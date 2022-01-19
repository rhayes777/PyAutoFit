from .condition import NameCondition, ValueCondition, StringValueCondition, TypeCondition
from .junction import And, Or
from .query import NamedQuery, Attribute, BooleanAttribute, ChildQuery
from .query.info import *


class N(NameCondition):
    pass


class V(ValueCondition):
    pass


class SV(StringValueCondition):
    pass


class T(TypeCondition):
    pass


class Q(NamedQuery):
    pass


class A(Attribute):
    pass


class BA(BooleanAttribute):
    pass
