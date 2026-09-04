#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .build import MODEL_REGISTRY, build_model  # noqa
from .pointformer import *  # noqa
from .query_class_matchability import install_query_class_matchability

# Install the Query-class confidence route after Pointformer is registered.
# The original builder remains active when the route is disabled.
install_query_class_matchability(Pointformer)
