"""Prepare Nexus's vendored ALF packaging for its Waymo/TensorFlow environment.

Only setup.py's install_requires list is changed. Runtime compatibility must
still be checked on the Linux GPU instance after installation.
"""

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path


REPLACEMENTS = {
    "absl-py": "absl-py==1.4.0",
    "gym": "gym==0.23.1",
    "matplotlib": "matplotlib==3.6.1",
    "numpy": "numpy==1.23.0",
    "protobuf": "protobuf==4.25.9",
    "tensorboard": "tensorboard==2.12.3",
    "gin-config": "gin-config==0.4.0",
    "cnest": "cnest==1.1.1",
    "opencv-python": "opencv-python-headless==4.9.0.80",
}

# These are the original setup's game, documentation and developer packages.
OMIT = {
    "atari-py", "box2d-py", "cpplint", "clang-format", "gym3", "pybullet",
    "sphinx", "sphinx-autobuild", "sphinx-autodoc-typehints",
    "sphinxcontrib-napoleon", "sphinx-rtd-theme",
}


def requirement_name(requirement):
    match = re.match(r"[A-Za-z0-9_.-]+", requirement)
    if match is None:
        raise ValueError("Unrecognized requirement: " + requirement)
    return re.sub(r"[-_.]+", "-", match.group()).lower()


def transform(source, headless=True):
    replacements = dict(REPLACEMENTS)
    if not headless:
        # Recognize the exact packaging emitted by the first script release.
        replacements.pop("opencv-python")
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "setup"]
    if len(calls) != 1:
        raise ValueError("Expected exactly one setup() call; nothing changed.")
    matches = [kw.value for kw in calls[0].keywords
               if kw.arg == "install_requires"]
    if len(matches) != 1:
        raise ValueError("Expected one install_requires list; nothing changed.")
    old = ast.literal_eval(matches[0])
    if not isinstance(old, list) or not all(isinstance(item, str) for item in old):
        raise ValueError("Unexpected dependency list; nothing changed.")
    found = {requirement_name(item) for item in old}
    missing = (set(replacements) | OMIT) - found
    if missing:
        raise ValueError("Upstream layout changed; missing: " + ", ".join(sorted(missing)))
    new = [replacements.get(requirement_name(item), item) for item in old
           if requirement_name(item) not in OMIT]
    segment = ast.get_source_segment(source, matches[0])
    if not segment or source.count(segment) != 1:
        raise ValueError("Cannot uniquely locate dependency list; nothing changed.")
    replacement = "[\n" + "".join("        " + repr(item) + ",\n" for item in new) + "    ]"
    result = source.replace(segment, replacement, 1)
    ast.parse(result)
    return result, old, new


def check_runtime(root):
    from importlib.metadata import version
    import cloudpickle
    import numpy as np
    import cv2
    import torch
    import gin
    import gym
    import alf
    from alf.environments import _penv, suite_gym

    expected = {
        "numpy": "1.23.0", "torch": "2.3.0+cu121",
        "torchvision": "0.18.0+cu121", "torchaudio": "2.3.0+cu121",
        "absl-py": "1.4.0", "matplotlib": "3.6.1",
        "protobuf": "4.25.9", "tensorboard": "2.12.3",
        "cloudpickle": "3.1.2", "gin-config": "0.4.0",
        "gym": "0.23.1", "cnest": "1.1.1",
        "opencv-python-headless": "4.9.0.80",
    }
    for package, wanted in expected.items():
        actual = version(package)
        if actual != wanted:
            raise RuntimeError(f"Unexpected version: {package}={actual}; wanted {wanted}")
        print(f"{package}: {actual}")
    if not Path(alf.__file__).resolve().is_relative_to(root / "third_party" / "alf"):
        raise RuntimeError("ALF was not imported from this Nexus checkout")
    if not Path(gin.__file__).resolve().is_relative_to(root / "third_party" / "gin-config"):
        raise RuntimeError("Gin was not imported from the cloned HorizonRobotics fork")
    print("ALF:", alf.__file__)
    print("C++ extension:", _penv.__file__)
    resized = cv2.resize(np.zeros((8, 8, 3), dtype=np.uint8), (4, 4))
    if resized.shape != (4, 4, 3):
        raise RuntimeError("OpenCV image resize check failed")
    print("OpenCV import and resize: OK")
    if alf.nest.flatten({"x": [1, 2]}) != [1, 2]:
        raise RuntimeError("ALF/cnest nested-structure check failed")
    if cloudpickle.loads(cloudpickle.dumps(lambda value: value + 1))(2) != 3:
        raise RuntimeError("cloudpickle round trip failed")

    class TinyEnvironment(gym.Env):
        observation_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        action_space = gym.spaces.Discrete(2)

        def reset(self, **kwargs):
            self.steps = 0
            return np.zeros(2, dtype=np.float32)

        def step(self, action):
            self.steps += 1
            return np.zeros(2, dtype=np.float32), 1.0, self.steps >= 2, {}

    environment = suite_gym.wrap_env(TinyEnvironment(), image_channel_first=False)
    try:
        first = environment.reset()
        following = environment.step(np.int64(0))
        if first.observation.shape != (2,) or following.observation.shape != (2,):
            raise RuntimeError("Gym/ALF observation shape check failed")
    finally:
        environment.close()
    print("Gym/ALF reset and step: OK")
    if not torch.cuda.is_available() or torch.version.cuda != "12.1":
        raise RuntimeError("Expected CUDA 12.1 PyTorch with an available GPU")
    values = torch.randn(256, 256, device="cuda")
    product = values @ values
    torch.cuda.synchronize()
    if not torch.isfinite(product).all().item():
        raise RuntimeError("GPU computation failed")
    print("ALF_RUNTIME_OK: imports, C++ extension, Gym adapter and GPU checks passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", type=Path, help="Path to the Nexus checkout")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--apply", action="store_true", help="Write the reviewed packaging adjustment")
    modes.add_argument("--check", action="store_true", help="Check the installed Linux runtime")
    args = parser.parse_args()
    root = args.repository.resolve(strict=True)
    if args.check:
        check_runtime(root)
        return
    target = root / "third_party" / "alf" / "setup.py"
    if target.is_symlink():
        raise SystemExit("Refusing a symlinked setup.py; nothing changed.")
    backup = target.with_name("setup.py.before-nexus-compat")
    current = target.read_bytes()
    original = backup.read_bytes() if backup.exists() else current
    result, before, after = transform(original.decode("utf-8"))
    encoded = result.encode("utf-8")
    previous, _, _ = transform(original.decode("utf-8"), headless=False)
    if current not in (original, encoded, previous.encode("utf-8")):
        raise SystemExit("setup.py has additional edits; nothing changed.")
    report = {
        "purpose": "Nexus core runtime; not a full ALF games/documentation installation",
        "original_sha256": hashlib.sha256(original).hexdigest(),
        "modified_sha256": hashlib.sha256(encoded).hexdigest(),
        "original_requirements": before,
        "nexus_requirements": after,
        "omitted_requirements": [item for item in before if requirement_name(item) in OMIT],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not args.apply:
        print("PREVIEW_ONLY: no files changed. Add --apply to write the adjustment.")
        return
    if not backup.exists():
        with backup.open("xb") as stream:
            stream.write(original)
    target.write_bytes(encoded)
    (root / "alf_compat_changes.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("ALF_PACKAGING_READY: backup saved; only install_requires was changed.")


if __name__ == "__main__":
    main()
