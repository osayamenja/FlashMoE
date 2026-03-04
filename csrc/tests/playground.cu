//
// Created by osayamen on 12/22/25.
//

// place to experiment
#include "debug.cuh"

struct Foo {
  int a;
  int b;
};
int main() {
  constexpr Foo foo{6, 7};
  const auto p = new Foo(foo);
  p->a += 1;
  auto q = *p;
  printf("a is %d\n", q.a);
  delete p;
}