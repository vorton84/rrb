# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.fsdp_workers import ActorRolloutRefWorker

from .ktsc_actor import KtscDataParallelPPOActor

class KtscActorRolloutRefWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        if getattr(self, "_is_actor", False) and getattr(self, "actor", None) is not None:
            base_name = type(self.actor).__name__
            assert base_name == "DataParallelPPOActor", (
                f"KTSC expected the worker to build DataParallelPPOActor, found {base_name}; "
                "refusing to swap __class__ on an unknown actor type"
            )
            self.actor.__class__ = KtscDataParallelPPOActor
            print(f"[ktsc] actor class swapped to {type(self.actor).__name__}")
