from __future__ import annotations

import math
import numpy as np
from scipy.cluster.hierarchy import DisjointSet
from typing import Callable, Dict, Set, Tuple, List, Final

from sortedcontainers import SortedSet


class Edge:
    def __init__(self, node1: int, node2: int,
                 weight: float):
        self.__node1 = min(node1, node2)
        self.__node2 = max(node1, node2)
        self.__weight = weight

    def __repr__(self) -> str:
        return f'Edge(weight={self.weight}, node1={self.node1}, node2={self.node2})'

    def __hash__(self) -> int:
        return hash((self.node1, self.node2))

    def __eq__(self, other: Edge) -> bool:
        if not isinstance(other, Edge):
            return False
        return self.node1 == other.node1 and self.node2 == other.node2

    # Per https://docs.python.org/3/library/functools.html#functools.total_ordering
    # Although cumbersome, below is more efficient than @functools.total_ordering
    def __ne__(self, other: Edge) -> bool:
        return not self == other

    def __lt__(self, other: Edge) -> bool:
        return (self.weight, self.node1, self.node2) < (other.weight, other.node1, other.node2)

    def __le__(self, other: Edge) -> bool:
        return (self.weight, self.node1, self.node2) <= (other.weight, other.node1, other.node2)

    def __gt__(self, other: Edge) -> bool:
        return (self.weight, self.node1, self.node2) > (other.weight, other.node1, other.node2)

    def __ge__(self, other: Edge) -> bool:
        return (self.weight, self.node1, self.node2) >= (other.weight, other.node1, other.node2)

    @property
    def node1(self) -> int:
        return self.__node1

    @property
    def node2(self) -> int:
        return self.__node2

    @property
    def weight(self) -> float:
        return self.__weight

    def get_neighbor(self, node: int) -> int:
        assert self.__node1 == node or self.__node2 == node
        return self.__node2 if node == self.node1 else self.__node1


class Worker:
    def __init__(self, worker_id: int, compute_flops: int, mem_capacity: int):
        self.__worker_id = worker_id
        self.__flops = compute_flops
        self.__memCapacity = mem_capacity

    @property
    def worker_id(self) -> int:
        return self.__worker_id

    @property
    def flops(self) -> int:
        return self.__flops

    @property
    def mem_capacity(self) -> int:
        return self.__memCapacity

    def __repr__(self) -> str:
        return f'Worker(id={self.worker_id}, flops={self.flops}, mem_capacity={self.mem_capacity})'

    def __hash__(self) -> int:
        return hash(self.worker_id)

    def __eq__(self, other: Worker) -> bool:
        if not isinstance(other, Worker):
            return False
        return self.worker_id == other.worker_id


class Group:
    TOTAL_COMPUTE: Final[str] = "totalCompute"
    TOTAL_EXPERT_WORKLOAD: Final[str] = "totalExpertWorkload"
    COMMUNICATION_FREQUENCY: Final[str] = "communicationFrequency"
    COMMUNICATION_COST: Final[str] = "communicationCost"
    RECOMPUTATION_AMOUNT: Final[str] = "recomputationAmount"
    GLOBAL_BATCH_SIZE: Final[str] = "globalBatchSize"
    NUM_LAYERS: Final[str] = "numLayers"
    MEM_CAPACITY: Final[str] = "memCapacity"
    MOE_FREQUENCY: Final[str] = "moeFrequency"
    EFFECTIVE_WORLD: Final[str] = "effectiveWorld"
    MINI_BATCH_SIZE: Final[str] = "miniBatchSize"
    NUM_GROUPS: Final[str] = "numGroups"
    NUM_EXPERTS: Final[str] = "numExperts"
    ALL_REDUCE_TIME: Final[str] = "allReduceTime"
    BOTTLENECK_TIME: Final[str] = "bottleneckTime"
    GAMMA: Final[str] = "gamma"
    RING_ALPHA: Final[str] = "ringAlpha"
    RING_BETA: Final[str] = "ringBeta"
    ALL_REDUCE_BUFFER: Final[str] = "allReduceBuffer"

    def __init__(self, group_id: int,
                 seed_node_compute: int,
                 external_edges: Dict[int, Edge],
                 obj: Callable[[Dict[str, float]], float],
                 all_reduce_func: Callable[[Dict[str, float]], float],
                 gamma: Callable[[Dict[str, float]], float],
                 p2p_time: float,
                 all_reduce_time: float,
                 expert_workload: int,
                 mem_capacity: int,
                 p2p_frequency: int,
                 num_experts: int,
                 world: int):
        self.groupID = group_id
        self.totalInternalEdgeWeights: Dict[int, float] = dict()
        self.memCapacity = mem_capacity
        self.externalEdges = external_edges
        self.objectiveFunction = obj
        self.allReduceFunc = all_reduce_func
        self.gamma = gamma
        self.p2pTime = p2p_time
        self.allReduceTime = all_reduce_time
        self.totalExpertFlops = expert_workload
        self.numExperts = num_experts
        self.totalComputeFLOPS = seed_node_compute
        self.eta = p2p_frequency
        self.currentObjective = self.objectiveFunction(self.__construct_obj_args(self.totalComputeFLOPS,
                                                                                 self.totalExpertFlops,
                                                                                 self.eta,
                                                                                 self.p2pTime,
                                                                                 self.memCapacity,
                                                                                 self.allReduceTime,
                                                                                 self.numExperts))

        self.cachedObjective = self.currentObjective
        self.cachedP2PTime = self.p2pTime
        self.cachedEffectiveWorld = 1 if self.memCapacity >= self.numExperts else 0
        self.world = world

    def evaluate_objective(self, internal_node: int, external_node: int,
                           neighbor_group: Group,
                           all_reduce_func_args: Dict[str, float],
                           gamma_args: Dict[str, int]) -> bool:
        # effective world
        effective_world = gamma_args[Group.EFFECTIVE_WORLD]
        if self.memCapacity + neighbor_group.memCapacity >= self.numExperts:
            if effective_world < self.world and self.memCapacity < self.numExperts:
                effective_world += 1
            if effective_world < self.world and neighbor_group.memCapacity < self.memCapacity:
                effective_world += 1

        gamma_args[Group.EFFECTIVE_WORLD] = effective_world
        self.cachedEffectiveWorld = effective_world
        all_reduce_func_args[Group.GAMMA] = self.gamma(gamma_args)

        self.allReduceTime = self.allReduceFunc(all_reduce_func_args)
        self.cachedP2PTime = self.evaluate_global_p2p_time(internal_node, external_node, neighbor_group)
        args = self.__construct_obj_args(total_compute_flops=self.totalComputeFLOPS + neighbor_group.totalComputeFLOPS,
                                         total_expert_flops=self.totalExpertFlops,
                                         communication_frequency=self.eta,
                                         communication_cost=self.cachedP2PTime,
                                         mem_capacity=self.memCapacity + neighbor_group.memCapacity,
                                         all_reduce_time=self.allReduceTime,
                                         num_experts=self.numExperts)
        self.cachedObjective = self.objectiveFunction(args)
        return self.cachedObjective < self.currentObjective

    def update_p2p_time(self):
        self.p2pTime = self.cachedP2PTime

    def update_all_reduce_time(self, all_reduce_time: float):
        self.allReduceTime = all_reduce_time
        self.currentObjective = self.objectiveFunction(self.__construct_obj_args(self.totalComputeFLOPS,
                                                                                 self.totalExpertFlops,
                                                                                 self.eta,
                                                                                 self.p2pTime,
                                                                                 self.memCapacity,
                                                                                 self.allReduceTime,
                                                                                 self.numExperts))

    def evaluate_global_p2p_time(self, internal_node: int, external_node: int, neighbor_group: Group):
        return max(self.p2pTime, self.totalInternalEdgeWeights.get(internal_node, 0) +
                   self.externalEdges[external_node].weight,
                   neighbor_group.evaluate_local_p2p_time(external_node, internal_node))

    def evaluate_local_p2p_time(self, internal_node: int, external_node: int) -> float:
        return max(self.p2pTime, self.totalInternalEdgeWeights.get(internal_node, 0) +
                   self.externalEdges[external_node].weight)

    def subsume_group(self, internal_node: int, external_node: int,
                      child: Group, groups: DisjointSet,
                      global_group_info: Dict[int, Group]) -> Tuple[Group, Set[Edge]]:
        self.update_p2p_time()
        self.currentObjective = self.cachedObjective
        self.memCapacity = self.memCapacity + child.memCapacity
        self.totalComputeFLOPS = self.totalComputeFLOPS + child.totalComputeFLOPS
        self.totalInternalEdgeWeights.update({internal_node: self.totalInternalEdgeWeights
                                             .get(internal_node, 0) + self.externalEdges[child.groupID].weight})
        self.totalInternalEdgeWeights.update({external_node: self.totalInternalEdgeWeights
                                             .get(external_node, 0) + self.externalEdges[child.groupID].weight})

        external_edges_by_group: Dict[int, Edge] = dict()
        external_edges_by_node: Dict[int, Edge] = dict()
        pruned_edges: Set[Edge] = set()
        group_parent = groups[internal_node]

        for ext_info in [(self.externalEdges, internal_node), (child.externalEdges, external_node)]:
            ext_edges = ext_info[0]
            current_node = ext_info[1]
            for node in ext_edges:
                updated_group = groups[node]
                if not groups.connected(group_parent, node):
                    ext_edge = external_edges_by_group.get(updated_group, None)
                    updated_ext_edge = ext_edges[node]
                    updated_node = updated_ext_edge.get_neighbor(current_node)
                    if ext_edge is not None:
                        pruned_edge = ext_edge
                        pruned_neighbor = pruned_edge.get_neighbor(internal_node)
                        if ext_edge.weight >= updated_ext_edge.weight:
                            pruned_edge = updated_ext_edge
                            updated_ext_edge = ext_edge
                            updated_node = updated_ext_edge.get_neighbor(internal_node)
                            pruned_neighbor = pruned_edge.get_neighbor(external_node)

                        pruned_edges.add(pruned_edge)
                        global_group_info[groups[pruned_neighbor]].externalEdges.pop(
                            pruned_edge.get_neighbor(pruned_neighbor))
                    external_edges_by_node.update({updated_node: updated_ext_edge})
                    external_edges_by_group.update({updated_group: updated_ext_edge})

        self.externalEdges = external_edges_by_group
        return self, pruned_edges

    @staticmethod
    def __construct_obj_args(total_compute_flops: int,
                             total_expert_flops: int,
                             communication_frequency: int,
                             communication_cost: float,
                             mem_capacity: int,
                             all_reduce_time: float,
                             num_experts: int) -> Dict[str, float]:
        return {Group.TOTAL_COMPUTE: total_compute_flops,
                Group.TOTAL_EXPERT_WORKLOAD: total_expert_flops,
                Group.COMMUNICATION_FREQUENCY: communication_frequency,
                Group.COMMUNICATION_COST: communication_cost,
                Group.MEM_CAPACITY: mem_capacity,
                Group.ALL_REDUCE_TIME: all_reduce_time,
                Group.NUM_EXPERTS: num_experts}

    @staticmethod
    def construct_all_reduce_args(gamma: float,
                                  num_groups: int,
                                  bottleneck_edge_time: float) -> Dict[str, float]:
        return {Group.GAMMA: gamma,
                Group.NUM_GROUPS: num_groups,
                Group.BOTTLENECK_TIME: bottleneck_edge_time}


def grigora2(a: np.ndarray, obj: Callable[[Dict[str, float]], float],
             all_reduce_func: Callable[[Dict[str, float]], float],
             gamma: Callable[[Dict[str, float]], float],
             p2p_buffer_size: int,
             p2p_freq: int,
             all_reduce_buffer_size: int,
             workers: List[Worker],
             expert_workload: list,
             gamma_args: Dict[str, int]) -> DisjointSet:
    lower_bound_num_groups_ar: Final[int] = 2  # why 2? lower bound argument
    invalid_groups: Set[Worker] = set()
    for worker in workers:
        if worker.mem_capacity < len(expert_workload):
            invalid_groups.add(worker)

    groups = DisjointSet(np.arange(a.shape[0]))
    global_candidate_edges = SortedSet()
    global_external_edges = SortedSet()
    group_info: Dict[int, Group] = dict()
    exp_workload = sum(expert_workload)

    gamma_args.update({Group.EFFECTIVE_WORLD: len(workers) - len(invalid_groups)})

    # We assume an undirected, simple graph.
    for i in range(a.shape[0]):
        group_external_edges: Dict[int, Edge] = dict()
        for j in range(a.shape[0]):
            if i != j and a[i][j][0] is not None:
                alpha: float = a.item((i, j, 0))
                beta: float = a.item((i, j, 1))
                p2p_edge = Edge(i, j, weight=p2p_transfer_time(alpha=alpha, beta=beta, buffer_size=p2p_buffer_size))
                global_candidate_edges.add(p2p_edge)
                group_external_edges.update({j: p2p_edge})
                global_external_edges.add(
                    Edge(i, j, weight=all_reduce_bottleneck_time(alpha=alpha,
                                                                 beta=beta, buffer_size=all_reduce_buffer_size,
                                                                 num_participants=lower_bound_num_groups_ar)))

        g = Group(group_id=i,
                  seed_node_compute=workers[i].flops,
                  external_edges=group_external_edges,
                  obj=obj,
                  all_reduce_func=all_reduce_func,
                  p2p_time=0,
                  all_reduce_time=0,  # deferred for later update
                  expert_workload=exp_workload,
                  p2p_frequency=p2p_freq,
                  mem_capacity=workers[i].mem_capacity,
                  num_experts=len(expert_workload),
                  world=len(workers),
                  gamma=gamma)
        group_info.update({i: g})

    init_ext_edge = global_external_edges[-1]
    b_t = all_reduce_bottleneck_time(alpha=a.item(init_ext_edge.node1, init_ext_edge.node2, 0),
                                     beta=a.item(init_ext_edge.node1, init_ext_edge.node2, 1),
                                     buffer_size=all_reduce_buffer_size,
                                     num_participants=(len(group_info) - len(invalid_groups)))
    a_r_args = Group.construct_all_reduce_args(gamma=gamma(gamma_args),
                                               num_groups=(len(group_info) - len(invalid_groups)),
                                               bottleneck_edge_time=b_t)
    a_r_time = all_reduce_func(a_r_args)
    for group_id in group_info:
        group_info[group_id].update_all_reduce_time(a_r_time)

    while len(global_candidate_edges) > 0:
        candidate_edge = global_candidate_edges.pop(0)
        n1 = candidate_edge.node1
        n2 = candidate_edge.node2
        group_n1 = groups[n1]
        group_n2 = groups[n2]

        e = global_external_edges[-1]
        a_r_args[Group.NUM_GROUPS] = len(group_info) - len(invalid_groups) - 1
        a_r_args[Group.BOTTLENECK_TIME] = all_reduce_bottleneck_time(alpha=a.item((e.node1, e.node2, 0)),
                                                                     beta=a.item((e.node1, e.node2, 1)),
                                                                     buffer_size=all_reduce_buffer_size,
                                                                     num_participants=a_r_args[Group.NUM_GROUPS])

        group1_approves_merge = group_info[group_n1].evaluate_objective(internal_node=n1,
                                                                        external_node=n2,
                                                                        neighbor_group=group_info[group_n2],
                                                                        all_reduce_func_args=a_r_args,
                                                                        gamma_args=gamma_args)
        group2_approves_merge = group_info[group_n2].evaluate_objective(internal_node=n2,
                                                                        external_node=n1,
                                                                        neighbor_group=group_info[group_n1],
                                                                        all_reduce_func_args=a_r_args,
                                                                        gamma_args=gamma_args)
        if group1_approves_merge and group2_approves_merge:
            groups.merge(group_n1, group_n2)
            if group_n1 == groups[group_n1]:
                parent_group = group_n1
                internal_n = n1
                external_n = n2
                child_group = group_n2
            else:
                parent_group = group_n2
                internal_n = n2
                external_n = n1
                child_group = group_n1

            merged_group, pruned_edges = group_info[parent_group].subsume_group(internal_node=internal_n,
                                                                                external_node=external_n,
                                                                                child=group_info[child_group],
                                                                                groups=groups,
                                                                                global_group_info=group_info)
            pruned_edges.add(candidate_edge) # already merged
            group_info.update({parent_group: merged_group})
            group_info.pop(child_group)

            if merged_group.memCapacity >= len(expert_workload):
                invalid_groups.discard(parent_group)
                invalid_groups.discard(child_group)
                gamma_args[Group.EFFECTIVE_WORLD] = min(len(workers), int(gamma_args[Group.EFFECTIVE_WORLD]) + 2)
            else:
                invalid_groups.discard(child_group)

            for edge in pruned_edges:
                global_candidate_edges.discard(edge)
                ext_alpha: float = a.item((edge.node1, edge.node2, 0))
                ext_beta: float = a.item((edge.node1, edge.node2, 1))
                global_external_edges.discard(
                    Edge(edge.node1, edge.node2,
                         weight=all_reduce_bottleneck_time(alpha=ext_alpha,
                                                           beta=ext_beta,
                                                           buffer_size=all_reduce_buffer_size,
                                                           num_participants=lower_bound_num_groups_ar)))
            if groups.connected(e.node1, e.node2):
                global_external_edges.discard(e)
        else:
            group_info[groups[group_n1]].externalEdges.pop(n2)
            group_info[groups[group_n2]].externalEdges.pop(n1)
    return groups


def p2p_transfer_time(alpha: float, beta: float, buffer_size: int) -> float:
    return alpha + (beta * buffer_size)


def all_reduce_bottleneck_time(alpha: float, beta: float, buffer_size: int, num_participants) -> float:
    if num_participants <= 0:
        return 0
    return alpha + (beta * (buffer_size / num_participants))


def gamma_function(args: Dict[str, float]):
    if args[Group.EFFECTIVE_WORLD] == 0:
        return math.inf
    num_moe_layers = args[Group.NUM_LAYERS] / args[Group.MOE_FREQUENCY]
    num_sequential_steps = args[Group.GLOBAL_BATCH_SIZE] / (args[Group.EFFECTIVE_WORLD] * args[Group.MINI_BATCH_SIZE])
    num_moe_executions_per_step = 2 + args[Group.RECOMPUTATION_AMOUNT]

    return num_moe_layers * num_sequential_steps * num_moe_executions_per_step


def all_reduce_function(args: Dict[str, float]) -> float:
    if math.isinf(args[Group.GAMMA]):
        return 0
    # Per Rolf Rabenseifner, https://link.springer.com/content/pdf/10.1007/978-3-540-24685-5_1.pdf
    all_reduce_time = 2 * (args[Group.NUM_GROUPS] - 1) * args[Group.BOTTLENECK_TIME]
    return all_reduce_time / args[Group.GAMMA]


def expert_parallel_group_objective_function(args: Dict[str, float]) -> float:
    if args[Group.MEM_CAPACITY] < args[Group.NUM_EXPERTS]:
        return math.inf

    return ((args[Group.TOTAL_EXPERT_WORKLOAD] / args[Group.TOTAL_COMPUTE]) +
            (args[Group.COMMUNICATION_FREQUENCY] * args[Group.COMMUNICATION_COST]) +
            args[Group.ALL_REDUCE_TIME])


if __name__ == '__main__':
    adjacency = np.array([[(None, None), (0, 1), (0, 2), (None, None), (None, None), (None, None)],
                          [(0, 1), (None, None), (None, None), (0, 3), (None, None), (None, None)],
                          [(0, 2), (None, None), (None, None), (None, None), (0, 5), (None, None)],
                          [(None, None), (0, 3), (None, None), (None, None), (None, None), (0, 2)],
                          [(None, None), (None, None), (0, 5), (None, None), (None, None), (0, 2)],
                          [(None, None), (None, None), (None, None), (0, 2), (0, 2), (None, None)]])

    adjacency = np.array([[(None, None), (0, 1), (0, 1), (0, 1)],
                          [(0, 1), (None, None), (0, 1), (0, 1)],
                          [(0, 1), (0, 1), (None, None), (0, 1)],
                          [(0, 1), (0, 1), (0, 1), (None, None)]])

    w = []
    for ii in range(adjacency.shape[0]):
        w.append(Worker(ii, 1, 12))
    n_exp = 12
    exp = []
    for ii in range(n_exp):
        exp.append(1)
    p2p_buf = 1
    p2p_fr = 4
    all_r_buf = 1
    gamma_arguments = {Group.NUM_LAYERS: 4,
                       Group.GLOBAL_BATCH_SIZE: 4,
                       Group.MINI_BATCH_SIZE: 2,
                       Group.MOE_FREQUENCY: 2,
                       Group.RECOMPUTATION_AMOUNT: 1}

    print(grigora2(a=adjacency,
                   obj=expert_parallel_group_objective_function,
                   all_reduce_func=all_reduce_function,
                   gamma=gamma_function,
                   p2p_buffer_size=p2p_buf,
                   p2p_freq=p2p_fr,
                   all_reduce_buffer_size=all_r_buf,
                   workers=w,
                   expert_workload=exp,
                   gamma_args=gamma_arguments).subsets())
