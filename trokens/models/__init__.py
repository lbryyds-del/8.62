#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .build import MODEL_REGISTRY, build_model  # noqa
from .pointformer import *  # noqa
from .query_class_matchability import install_query_class_matchability

# Install the optional 8.64 extension after Pointformer is registered. The
# original builder remains active whenever QUERY_CLASS_MATCHABILITY is disabled.
install_query_class_matchability(Pointformer)
