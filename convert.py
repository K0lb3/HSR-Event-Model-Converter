import os
import re
from converter import SceneConverter


def main(work_dir: str = None, avatar: int = -1):
    work_dir = work_dir or input("Enter work dir: ")
    avatars = [
        match.group(1)
        for match in (re.match(r"avatar_(.+?)_model", f) for f in os.listdir(work_dir))
        if match is not None
    ]
    if isinstance(avatar, int) and avatar == -1:
        print("Avatars:")
        print(f"\t0 - All")
        for i, avatar in enumerate(avatars):
            print(f"\t{i + 1} - {avatar}")
        avatar = int(input("Enter avatar: "))

    if avatar == 0:
        for avatar in avatars:
            convert_avatar(work_dir, avatar)
    else:
        convert_avatar(work_dir, avatars[avatar - 1])


def convert_avatar(work_dir: str, avatar: str):
    conv = SceneConverter(work_dir, avatar)
    # hotfix colors
    for mesh in conv.gltf.meshes:
        for prim in mesh.primitives:
            attr = prim.attributes
            attr.COLOR_0 = None
            attr.COLOR_1 = None
    conv.gltf.save(f"{avatar}.gltf")


if __name__ == "__main__":
    main()
