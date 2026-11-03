"""에이전트 자동 등록.

agents/ 폴더 안의 모든 .py 파일을 읽어, Agent를 상속하고 name이 정해진 클래스를 AGENTS에 등록합니다.
새 에이전트는 파일만 추가하면 됩니다:  agents/my_agent.py 의  class MyAgent(Agent): name = "my_agent"
    -> python train.py --agent my_agent
"""

import importlib
import pkgutil
import sys
import traceback

from .base import Agent

AGENTS = {}

for _info in pkgutil.iter_modules(__path__):
    if _info.name.startswith("_") or _info.name == "base":
        continue
    try:
        _module = importlib.import_module(f"{__name__}.{_info.name}")
    except Exception:
        # 파일 하나에 오류가 있어도 다른 에이전트는 쓸 수 있게 경고만 출력
        print(f"[agents] 경고: agents/{_info.name}.py를 불러오지 못해 건너뜁니다.\n{traceback.format_exc()}",
              file=sys.stderr)
        continue
    for _obj in vars(_module).values():
        if (isinstance(_obj, type) and issubclass(_obj, Agent) and _obj is not Agent
                and _obj.__module__ == _module.__name__ and getattr(_obj, "name", "base") != "base"):
            if _obj.name in AGENTS and AGENTS[_obj.name] is not _obj:
                raise ValueError(f"에이전트 이름 '{_obj.name}'이(가) 중복됩니다: "
                                 f"{AGENTS[_obj.name].__module__} 와 {_obj.__module__}. "
                                 f"새 클래스에 고유한 name = \"...\"을 지정하세요 (상속하면 부모의 name을 물려받습니다).")
            AGENTS[_obj.name] = _obj
