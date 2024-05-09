from __future__ import annotations

import math
import numpy as np
from typing import Callable, Dict, Set, Tuple, List, Final

from scipy.cluster.hierarchy import DisjointSet
from sortedcontainers import SortedSet
from lazy_streams import stream


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
        assert isinstance(other, Edge)
        return not self == other

    def __lt__(self, other: Edge) -> bool:
        assert isinstance(other, Edge)
        return (self.weight, self.node1, self.node2) < (other.weight, other.node1, other.node2)

    def __le__(self, other: Edge) -> bool:
        assert isinstance(other, Edge)
        return (self.weight, self.node1, self.node2) <= (other.weight, other.node1, other.node2)

    def __gt__(self, other: Edge) -> bool:
        assert isinstance(other, Edge)
        return (self.weight, self.node1, self.node2) > (other.weight, other.node1, other.node2)

    def __ge__(self, other: Edge) -> bool:
        assert isinstance(other, Edge)
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

    def __ne__(self, other: Worker) -> bool:
        assert isinstance(other, Worker)
        return not self == other

    def __lt__(self, other: Worker) -> bool:
        assert isinstance(other, Worker)
        return self.flops < other.flops

    def __le__(self, other: Worker) -> bool:
        assert isinstance(other, Worker)
        return self.flops <= other.flops

    def __gt__(self, other: Worker) -> bool:
        assert isinstance(other, Worker)
        return self.flops > other.flops

    def __ge__(self, other: Worker) -> bool:
        assert isinstance(other, Worker)
        return self.flops >= other.flops


class Expert:
    def __init__(self, compute_cost: int, expert_id: int = None):
        self.__computeCost = compute_cost
        if expert_id is None:
            self.__expert_id = 0
        else:
            self.__expert_id = expert_id

    @property
    def expert_id(self) -> int:
        return self.__expert_id

    @property
    def compute_cost(self) -> int:
        return self.__computeCost

    def cost_most_similar_to(self, exps: List[Expert]) -> Expert:
        e = exps[0]
        min_dist = int(math.fabs(self.compute_cost - e.compute_cost))
        for i in range(1, len(exps)):
            dist = int(math.fabs(self.compute_cost - exps[i].compute_cost))
            if dist < min_dist:
                min_dist = dist
                e = exps[i]
        return e

    def __hash__(self) -> int:
        return hash((self.compute_cost, self.expert_id))

    def __repr__(self) -> str:
        return f'Expert(id={self.expert_id}, compute_cost={self.compute_cost})'

    def __eq__(self, other: Expert) -> bool:
        if not isinstance(other, Expert):
            return False
        return self.compute_cost == other.compute_cost and self.expert_id == other.expert_id

    def __ne__(self, other: Expert) -> bool:
        return not self == other

    def __lt__(self, other: Expert) -> bool:
        assert isinstance(other, Expert)
        return (self.compute_cost, self.expert_id) < (other.compute_cost, other.expert_id)

    def __le__(self, other: Expert) -> bool:
        assert isinstance(other, Expert)
        return (self.compute_cost, self.expert_id) <= (other.compute_cost, other.expert_id)

    def __gt__(self, other: Expert) -> bool:
        assert isinstance(other, Expert)
        return (self.compute_cost, self.expert_id) > (other.compute_cost, other.expert_id)

    def __ge__(self, other: Expert) -> bool:
        assert isinstance(other, Expert)
        return (self.compute_cost, self.expert_id) >= (other.compute_cost, other.expert_id)


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
                 obj: Callable[[Dict[str, float]], float],
                 all_reduce_func: Callable[[Dict[str, float]], float],
                 gamma: Callable[[Dict[str, float]], float],
                 p2p_time: float,
                 all_reduce_time: float,
                 expert_workload: int,
                 mem_capacity: int,
                 p2p_frequency: int,
                 num_experts: int,
                 world: int,
                 gamma_val: float):
        self.groupID = group_id
        self.internalP2PTimes: Dict[int, Tuple[float, float]] = {group_id: (0.0, 0.0)}
        self.memCapacity = mem_capacity
        self.objectiveFunction = obj
        self.allReduceFunc = all_reduce_func
        self.gamma = gamma
        self.p2pTime = p2p_time
        self.allReduceTime = all_reduce_time
        self.totalExpertFlops = expert_workload
        self.numExperts = num_experts
        self.totalComputeFLOPS = seed_node_compute
        self.eta = p2p_frequency
        self.currentObjective = self.objectiveFunction(self.__construct_obj_args(gamma_val,
                                                                                 self.totalComputeFLOPS,
                                                                                 self.totalExpertFlops,
                                                                                 self.eta,
                                                                                 self.p2pTime,
                                                                                 self.memCapacity,
                                                                                 self.allReduceTime,
                                                                                 self.numExperts))

        self.cachedObjective = self.currentObjective
        self.cachedP2PTime: Dict[int, Tuple[float, float]] = dict()
        self.cachedEffectiveWorld = 1 if self.memCapacity >= self.numExperts else 0
        self.world = world

    def num_nodes(self):
        return len(self.internalP2PTimes)

    def evaluate_objective(self, neighbor_group: Group, all_reduce_func_args: Dict[str, float],
                           gamma_args: Dict[str, int], adj: np.ndarray, p2p_buffer: int) -> bool:
        prev_effective_world = gamma_args[Group.EFFECTIVE_WORLD]
        effective_world = prev_effective_world
        if self.memCapacity + neighbor_group.memCapacity >= self.numExperts:
            if effective_world < self.world and self.memCapacity < self.numExperts:
                effective_world += self.num_nodes()
            if effective_world < self.world and neighbor_group.memCapacity < self.numExperts:
                effective_world += neighbor_group.num_nodes()

        gamma_args[Group.EFFECTIVE_WORLD] = effective_world
        self.cachedEffectiveWorld = effective_world
        self.allReduceTime = self.allReduceFunc(all_reduce_func_args)

        n_nodes = self.num_nodes() + neighbor_group.num_nodes()
        global_p2p_time = self.evaluate_global_p2p_time(neighbor_group,
                                                        n_nodes=n_nodes,
                                                        adj=adj,
                                                        p2p_buffer=p2p_buffer)
        args = self.__construct_obj_args(gamma=self.gamma(gamma_args),
                                         total_compute_flops=self.totalComputeFLOPS + neighbor_group.totalComputeFLOPS,
                                         total_expert_flops=self.totalExpertFlops,
                                         communication_frequency=self.eta,
                                         communication_cost=global_p2p_time,
                                         mem_capacity=self.memCapacity + neighbor_group.memCapacity,
                                         all_reduce_time=self.allReduceTime,
                                         num_experts=self.numExperts)
        self.cachedObjective = self.objectiveFunction(args)
        gamma_args[Group.EFFECTIVE_WORLD] = prev_effective_world  # restore global state
        if math.isinf(self.currentObjective):
            return True
        return self.cachedObjective < self.currentObjective

    def update_p2p_time(self, child: Group):
        for nodes in [self.internalP2PTimes, child.internalP2PTimes]:
            for node in nodes:
                self.internalP2PTimes.update({node: self.cachedP2PTime[node]})

    def evaluate_global_p2p_time(self, neighbor_group: Group, adj: np.ndarray, n_nodes: int, p2p_buffer: int) -> float:
        max_p2p_time = 0.0
        for nodes in [self.internalP2PTimes, neighbor_group.internalP2PTimes]:
            for node in nodes:
                self.cachedP2PTime.update({node: nodes[node]})

        for parent_node in self.internalP2PTimes:
            for child_node in neighbor_group.internalP2PTimes:
                details = self.cachedP2PTime[parent_node]
                p_alpha_sum = details[0] + adj.item(parent_node, child_node, 0)
                p_beta_sum = details[1] + adj.item(parent_node, child_node, 1)
                self.cachedP2PTime[parent_node] = (p_alpha_sum, p_beta_sum)

                details = self.cachedP2PTime[child_node]
                c_alpha_sum = details[0] + adj.item(parent_node, child_node, 0)
                c_beta_sum = details[1] + adj.item(parent_node, child_node, 1)
                self.cachedP2PTime[child_node] = (c_alpha_sum, c_beta_sum)
                split_buf = p2p_buffer / n_nodes
                max_p2p_time = max(max_p2p_time,
                                   p2p_transfer_time(p_alpha_sum, p_beta_sum, buffer_size=split_buf),
                                   p2p_transfer_time(c_alpha_sum, c_beta_sum, buffer_size=split_buf))

        return max_p2p_time

    def subsume_group(self, child: Group) -> Group:
        self.update_p2p_time(child)
        self.currentObjective = self.cachedObjective
        self.memCapacity = self.memCapacity + child.memCapacity
        self.totalComputeFLOPS = self.totalComputeFLOPS + child.totalComputeFLOPS
        return self

    @staticmethod
    def __construct_obj_args(gamma: float,
                             total_compute_flops: int,
                             total_expert_flops: int,
                             communication_frequency: int,
                             communication_cost: float,
                             mem_capacity: int,
                             all_reduce_time: float,
                             num_experts: int) -> Dict[str, float]:
        return {Group.GAMMA: gamma,
                Group.TOTAL_COMPUTE: total_compute_flops,
                Group.TOTAL_EXPERT_WORKLOAD: total_expert_flops,
                Group.COMMUNICATION_FREQUENCY: communication_frequency,
                Group.COMMUNICATION_COST: communication_cost,
                Group.MEM_CAPACITY: mem_capacity,
                Group.ALL_REDUCE_TIME: all_reduce_time,
                Group.NUM_EXPERTS: num_experts}

    @staticmethod
    def construct_all_reduce_args(num_groups: int, bottleneck_edge_time: float) -> Dict[str, float]:
        return {Group.NUM_GROUPS: num_groups, Group.BOTTLENECK_TIME: bottleneck_edge_time}


def grigora2(a: np.ndarray, obj: Callable[[Dict[str, float]], float],
             all_reduce_func: Callable[[Dict[str, float]], float],
             gamma: Callable[[Dict[str, float]], float],
             p2p_buffer_size: int,
             p2p_freq: int,
             all_reduce_buffer_size: int,
             workers: List[Worker],
             expert_workload: list,
             gamma_args: Dict[str, int]) -> Tuple[DisjointSet, Set[int]]:
    lower_bound_num_groups_ar: Final[int] = 2  # why 2? lower bound argument
    invalid_groups: Set[int] = set()
    for worker in workers:
        if worker.mem_capacity < len(expert_workload):
            invalid_groups.add(worker.worker_id)

    groups = DisjointSet(np.arange(a.shape[0]))
    global_candidate_edges = SortedSet()
    global_external_edges = SortedSet()
    group_info: Dict[int, Group] = dict()
    exp_workload = sum(expert_workload)

    gamma_args.update({Group.EFFECTIVE_WORLD: len(workers) - len(invalid_groups)})

    # Upper triangular traversal completes in time proportional to n*(n-1)/ 2 rather than n^2
    for i in range(a.shape[0]):
        for j in range(i + 1, a.shape[0]):
            if i != j:
                alpha: float = a.item((i, j, 0))
                beta: float = a.item((i, j, 1))
                p2p_edge = Edge(i, j, weight=p2p_transfer_time(alpha=alpha, beta=beta, buffer_size=p2p_buffer_size))
                global_candidate_edges.add(p2p_edge)
                global_external_edges.add(
                    Edge(i, j, weight=all_reduce_bottleneck_time(alpha=alpha,
                                                                 beta=beta, buffer_size=all_reduce_buffer_size,
                                                                 num_participants=lower_bound_num_groups_ar)))

    init_ext_edge = global_external_edges[-1]
    n_groups = len(workers) - len(invalid_groups)
    b_t = all_reduce_bottleneck_time(alpha=a.item(init_ext_edge.node1, init_ext_edge.node2, 0),
                                     beta=a.item(init_ext_edge.node1, init_ext_edge.node2, 1),
                                     buffer_size=all_reduce_buffer_size,
                                     num_participants=n_groups)
    a_r_args = Group.construct_all_reduce_args(num_groups=n_groups,
                                               bottleneck_edge_time=b_t)
    a_r_time = all_reduce_func(a_r_args)
    gamma_init = gamma(gamma_args)
    for i in range(len(workers)):
        group_info.update({i: Group(group_id=i,
                                    seed_node_compute=workers[i].flops,
                                    obj=obj,
                                    all_reduce_func=all_reduce_func,
                                    p2p_time=0,
                                    all_reduce_time=a_r_time,
                                    expert_workload=exp_workload,
                                    p2p_frequency=p2p_freq,
                                    mem_capacity=workers[i].mem_capacity,
                                    num_experts=len(expert_workload),
                                    world=len(workers),
                                    gamma=gamma,
                                    gamma_val=gamma_init)})

    while len(global_candidate_edges) > 0:
        candidate_edge = global_candidate_edges.pop(0)
        if not groups.connected(candidate_edge.node1, candidate_edge.node2):
            n1 = candidate_edge.node1
            n2 = candidate_edge.node2
            group_n1 = groups[n1]
            group_n2 = groups[n2]

            e = global_external_edges[-1]
            while groups.connected(e.node1, e.node2):
                global_external_edges.pop()
                e = global_external_edges[-1]

            n_groups = len(group_info) - len(invalid_groups)
            if group_n1 in invalid_groups and group_n2 in invalid_groups:
                if group_info[group_n1].memCapacity + group_info[group_n2].memCapacity >= len(expert_workload):
                    n_groups += 1
            elif group_n1 not in invalid_groups and group_n2 not in invalid_groups:
                n_groups -= 1

            a_r_args[Group.NUM_GROUPS] = n_groups
            a_r_args[Group.BOTTLENECK_TIME] = all_reduce_bottleneck_time(alpha=a.item((e.node1, e.node2, 0)),
                                                                         beta=a.item((e.node1, e.node2, 1)),
                                                                         buffer_size=all_reduce_buffer_size,
                                                                         num_participants=a_r_args[Group.NUM_GROUPS])

            group1_approves_merge = group_info[group_n1].evaluate_objective(neighbor_group=group_info[group_n2],
                                                                            all_reduce_func_args=a_r_args,
                                                                            gamma_args=gamma_args,
                                                                            adj=a,
                                                                            p2p_buffer=p2p_buffer_size)
            group2_approves_merge = group_info[group_n2].evaluate_objective(neighbor_group=group_info[group_n1],
                                                                            all_reduce_func_args=a_r_args,
                                                                            gamma_args=gamma_args,
                                                                            adj=a,
                                                                            p2p_buffer=p2p_buffer_size)
            if group1_approves_merge and group2_approves_merge:
                groups.merge(group_n1, group_n2)
                if group_n1 == groups[group_n1]:
                    parent_group = group_n1
                    child_group = group_n2
                else:
                    parent_group = group_n2
                    child_group = group_n1

                effective_world_size = gamma_args[Group.EFFECTIVE_WORLD]
                if group_info[parent_group].memCapacity + group_info[child_group].memCapacity >= len(expert_workload):
                    if parent_group in invalid_groups:
                        invalid_groups.discard(parent_group)
                        effective_world_size += group_info[parent_group].num_nodes()
                    if child_group in invalid_groups:
                        invalid_groups.discard(child_group)
                        effective_world_size += group_info[child_group].num_nodes()
                else:
                    invalid_groups.discard(child_group)
                gamma_args[Group.EFFECTIVE_WORLD] = effective_world_size
                merged_group = group_info[parent_group].subsume_group(child=group_info[child_group])
                group_info.update({parent_group: merged_group})
                group_info.pop(child_group)
    return groups, invalid_groups


def p2p_transfer_time(alpha: float, beta: float, buffer_size: float) -> float:
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
    # Per Rolf Rabenseifner, https://link.springer.com/content/pdf/10.1007/978-3-540-24685-5_1.pdf
    all_reduce_time = 2 * (args[Group.NUM_GROUPS] - 1) * args[Group.BOTTLENECK_TIME]
    return all_reduce_time


def expert_parallel_group_objective_function(args: Dict[str, float]) -> float:
    if args[Group.MEM_CAPACITY] < args[Group.NUM_EXPERTS]:
        return math.inf

    return args[Group.GAMMA] * ((args[Group.TOTAL_EXPERT_WORKLOAD] / args[Group.TOTAL_COMPUTE]) +
                                (args[Group.COMMUNICATION_FREQUENCY] * args[Group.COMMUNICATION_COST]) +
                                args[Group.ALL_REDUCE_TIME])


def match(budget: Expert, ss: SortedSet) -> Expert:
    if budget in ss:
        ss.discard(budget)
        return budget
    floor = ss.bisect_left(budget) - 1
    ceiling = ss.bisect_right(budget)
    if floor >= 0 and ceiling < len(ss):
        most_similar_cost = budget.cost_most_similar_to([ss[floor], ss[ceiling]])
        ss.discard(most_similar_cost)
        return most_similar_cost
    if floor >= 0:
        most_similar_cost = ss[floor]
    else:
        most_similar_cost = ss[ceiling]
    ss.discard(most_similar_cost)
    return most_similar_cost


def expert_assignment(workers: List[Worker], experts: List[Expert]) -> Dict[Worker, Set[int]]:
    mem_capacity: Dict[Worker, int] = dict()
    for worker1 in workers:
        mem_capacity.update({worker1: worker1.mem_capacity})
    assert stream(workers).map(lambda worker: worker.mem_capacity).reduce(lambda m1, m2: m1 + m2) >= len(experts)
    n_workers = len(workers)
    n_experts = len(experts)
    s: Dict[Worker, Set[int]] = dict()
    for w in workers:
        s.update({w: set()})

    # Determination of singleton multi-sets
    skip_sort = len(set(workers)) == 1 and len(set(experts)) == 1

    if n_experts % n_workers == 0:
        req_capacity = n_experts / n_workers
        each_worker_has_capacity = (stream(workers).
                                    filter(lambda worker: worker.mem_capacity >= req_capacity).size() == len(workers))
    else:
        req_capacity = math.ceil(n_experts / n_workers)
        outlying_req_capacity = (n_experts - ((n_workers - 1) * req_capacity))
        each_worker_has_capacity = ((stream(workers).take(len(workers) - 1).
                                     filter(lambda worker: worker.mem_capacity >= req_capacity))
                                    and workers[len(workers) - 1].mem_capacity >= outlying_req_capacity)
    if skip_sort:
        i = 0
        current_exp = 0
        remaining_workers = n_workers
        while current_exp < n_experts:
            budget = int(math.ceil((n_experts - current_exp) / remaining_workers))
            picked_worker = workers[i]
            m_w = mem_capacity[picked_worker]
            while budget > 0 and m_w > 0:
                expert = experts[current_exp]
                s.get(picked_worker).add(expert.expert_id)
                m_w -= 1
                budget -= 1
                current_exp += 1
            mem_capacity[picked_worker] = m_w
            i = (i + 1) % n_workers
            if m_w == 0 or each_worker_has_capacity:
                remaining_workers -= 1
        return s

    workers.sort(reverse=True)
    expert_compute_cost: SortedSet = SortedSet(experts)
    sum_w_flops = stream(workers).map(lambda worker: worker.flops).reduce(lambda flop1, flop2: flop1 + flop2)
    sum_e_cost = stream(experts).map(lambda x: x.compute_cost).reduce(lambda cost1, cost2: cost1 + cost2)
    i = 0
    while len(expert_compute_cost) > 0:
        picked_worker: Worker = workers[i]
        m_w = mem_capacity[picked_worker]
        budget = int(math.ceil((picked_worker.flops * sum_e_cost) / sum_w_flops))
        allocated_budget = budget
        while budget > 0 and m_w > 0 and len(expert_compute_cost) > 0:
            expert = match(Expert(budget), expert_compute_cost)
            s.get(picked_worker).add(expert.expert_id)
            m_w -= 1
            budget -= expert.compute_cost
        i = (i + 1) % n_workers
        mem_capacity[picked_worker] = m_w
        sum_e_cost -= (allocated_budget - budget)
        if m_w == 0 or each_worker_has_capacity:
            sum_w_flops -= picked_worker.flops
    return s


if __name__ == '__main__':
    dim = 16
    intra_node_width = 4.0

    adjacency = np.zeros((dim, dim, 2))
    intra_node_cost = (0.009, 0.014)  # (ms, ms/MB)
    inter_node_cost = (0.03, 0.054)

    for ii in range(adjacency.shape[0]):
        for jj in range(adjacency.shape[0]):
            if ii != jj:
                if math.floor(jj / intra_node_width) == math.floor(ii / intra_node_width):
                    # intra-node
                    adjacency[ii, jj] = intra_node_cost
                else:
                    # inter-node
                    adjacency[ii, jj] = inter_node_cost

    a100_theoretical_flop_per_ms = 312 * 1E9
    realistic_scaling_factor = 0.43
    real_flops = int(math.ceil(realistic_scaling_factor * a100_theoretical_flop_per_ms))

    mem = 32
    g_workers = []
    for ii in range(adjacency.shape[0]):
        g_workers.append(Worker(ii, real_flops, mem))

    n_exp = 64
    exp = []
    ml_experts = []
    exp_flops = 16 * 4 * 2048 * (1024 ** 2)
    for ii in range(n_exp):
        exp.append(exp_flops)
        ml_experts.append(Expert(exp_flops, ii))

    p2p_buf_mb = 16
    p2p_fr = 4
    all_r_buf = 512

    gamma_arguments = {Group.NUM_LAYERS: 24,
                       Group.GLOBAL_BATCH_SIZE: 256,
                       Group.MINI_BATCH_SIZE: 4,
                       Group.MOE_FREQUENCY: 2,
                       Group.RECOMPUTATION_AMOUNT: 1}

    shard_spec, inv = grigora2(a=adjacency,
                               obj=expert_parallel_group_objective_function,
                               all_reduce_func=all_reduce_function,
                               gamma=gamma_function,
                               p2p_buffer_size=p2p_buf_mb,
                               p2p_freq=p2p_fr,
                               all_reduce_buffer_size=all_r_buf,
                               workers=g_workers,
                               expert_workload=exp,
                               gamma_args=gamma_arguments)
    print(shard_spec.subsets())
    # experts_list: List[Expert] = [Expert(3, 0),
    #                               Expert(3, 1),
    #                               Expert(3, 2),
    #                               Expert(2, 3),
    #                               Expert(2, 4),
    #                               Expert(3, 5),
    #                               Expert(2, 6),
    #                               Expert(3, 7)]
    # workers_list: List[Worker] = [Worker(0, 42, 1),
    #                               Worker(1, 56, 5),
    #                               Worker(2, 68, 1),
    #                               Worker(3, 22, 3)]
    # print(expert_assignment(workers_list, experts_list))
