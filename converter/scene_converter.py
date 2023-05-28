import json
import os
import re
import struct
from typing import List, Tuple

import pygltflib

from .model_decoder import parse_model
from .utils import (
    euler_rotation_to_quaternion,
    flatten,
    get_image_mimetype,
    next_filter,
)

ATTRIBUTE_TO_GLTF = {
    "position": "POSITION",
    "normal": "NORMAL",
    "uv": "TEXCOORD_0",
    "uv2": "TEXCOORD_1",
    "color": "COLOR_0",
    "color2": "COLOR_1",
    "skinIndex": "JOINTS_0",
    "skinWeight": "WEIGHTS_0",
}


class SceneConverter:
    gltf: pygltflib.GLTF2
    unique_dict = {}
    scene: dict
    model: dict
    working_dir: str
    working_dir_files: List[str]
    node_lookup: dict

    def __init__(self, model_dir: str, avatar_name: str) -> None:
        self.gltf = pygltflib.GLTF2()
        self.working_dir = model_dir
        self.working_dir_files = os.listdir(model_dir)
        self.node_lookup = {}

        model_bin_f, scene_json_f, timeline_json_f = self.find_avatar_files(avatar_name)

        with open(os.path.join(model_dir, model_bin_f), "rb") as f:
            self.model = parse_model(f)
        with open(os.path.join(model_dir, scene_json_f), "rt", encoding="utf8") as f:
            self.scene = json.load(f)
            self.parse_scene(self.scene)
        with open(os.path.join(model_dir, timeline_json_f), "rt", encoding="utf8") as f:
            self.timelines = json.load(f)
            self.parse_timelines(self.timelines)

        self.gltf.scene = 0
        self.gltf.scenes.append(pygltflib.Scene(nodes=[0]))
        self.gltf.buffers.append(
            pygltflib.Buffer(
                byteLength=len(self.gltf._glb_data),
            )
        )

    def find_avatar_files(self, avatar_name: str) -> Tuple[str, str, str]:
        avatar_name_l = avatar_name.lower()
        model_bin_name = next_filter(
            self.working_dir_files,
            lambda f: re.match(f"avatar_{avatar_name_l}_model.+?.bin", f),
        )
        scene_json_name = next_filter(
            self.working_dir_files,
            lambda f: re.match(f"avatar_{avatar_name_l}_scene.+?.json", f),
        )
        timeline_json_name = next_filter(
            self.working_dir_files,
            lambda f: re.match(f"avatar_{avatar_name_l}_timeline.+", f),
        )
        return model_bin_name, scene_json_name, timeline_json_name

    def parse_timelines(self, timelines: dict):
        for key, timeline in timelines.items():
            self.parse_timeline(timeline, key)

    def parse_timeline(self, timeline: dict, name: str = None):
        # type int
        # start int
        # name str
        # loop bool
        # clips
        animation = pygltflib.Animation(name=name, channels=[], samplers=[])
        self.gltf.animations.append(animation)

        for clip in timeline["clips"]:
            channel, sampler = self.parse_timeline_clip(clip)
            if channel is None:
                continue
            channel.sampler = len(animation.samplers)
            animation.channels.append(channel)
            animation.samplers.append(sampler)

    def parse_timeline_clip(
        self, clip: dict
    ) -> Tuple[pygltflib.AnimationChannel, pygltflib.AnimationSampler]:
        # type int
        # start int
        # name str (node.position/quaternion/scale)
        # loop bool
        # valueType str (v3, q)
        # data
        # clipStart int
        # clipEnd int
        name, property_name = clip["name"].split(".")

        node_id = self.node_lookup.get(name)
        if node_id is None:
            print(
                f"Warning: Doesn't know how to handle {name} with {property_name} animation!"
            )
            return None, None
        if property_name == "position":
            target_path = "translation"
        elif property_name == "quaternion":
            target_path = "rotation"
        elif property_name == "scale":
            target_path = "scale"
        elif property_name == "morphTargetInfluences":
            target_path = "weights"
        else:
            raise ValueError(f"Unknown property {property_name}")

        channel = pygltflib.AnimationChannel(
            sampler=0,
            target=pygltflib.AnimationChannelTarget(
                node=node_id,
                path=target_path,
            ),
        )
        sampler = pygltflib.AnimationSampler(
            input=0,
            interpolation="LINEAR",
            output=1,
        )

        times = []
        values = []
        for item in clip["data"]:
            start = item["start"] / 60
            end = item["end"] / 60
            frames = item["frames"]
            if len(frames) == 1:
                times.append(start)
                values.append(frames[0])
            else:
                # assume equidistant frames
                equi_dist = (end - start) / (len(frames) - 1)
                times.extend([i * equi_dist + start for i in range(len(frames))])
                values.extend(frames)
            # might not be necessary?
            # but ensures that we stay within the specified frame animation
            times.append(end)
            values.append(frames[-1])

        times_bin = struct.pack(f"<{len(times)}f", *times)
        values_bin = struct.pack(f"<{len(values)*len(values[0])}f", *flatten(values))

        sampler.input = self.add_array(
            1,
            1,
            memoryview(times_bin).cast("f"),
            f"{name}_{property_name}_times",
            0,
        )
        sampler.output = self.add_array(
            len(values[0]) if property_name != "morphTargetInfluences" else 1,
            2,
            memoryview(values_bin).cast("f"),
            f"{name}_{property_name}_values",
            0,
        )

        return channel, sampler

    def parse_scene(self, scene: dict):
        # layoutData -> scene structure and camera
        # imgConfigData -> images with wrapMode - will be used when registering textures
        layout = scene["layoutData"]
        # sceneConfig - uiWidth, uiHeight - can be ignored
        # camera - ignored for now
        # hooksFunc - initial animation?
        # children - actual scene, has two root children, ground and avatar
        #    ignore ground for now
        avatar = next(
            child for child in layout["children"] if child["name"] == "AVATAR"
        )
        self.parse_scene_node(avatar)

        # patch skinning -> skins.joints (str) -> (int)
        for skin in self.gltf.skins:
            skin.joints = [self.node_lookup[bone["name"]] for bone in skin.joints]

    def parse_scene_node(self, item: dict) -> int:
        # type
        #    0 - node
        #    2 - mesh
        #    6-  joint/bone
        #    7 - Camera
        # name
        # uuid
        # visible
        # renderOrder
        # children
        uuid = item.get("uuid")
        if uuid in self.unique_dict:
            return self.unique_dict[uuid]

        node_id = len(self.gltf.nodes)
        self.unique_dict[uuid] = node_id
        node = pygltflib.Node(
            name=item.get("name"),
            matrix=item.get("matrix"),
            translation=item.get("position"),
            rotation=euler_rotation_to_quaternion(item.get("rotation")),
            scale=item.get("scale"),
        )

        if item["type"] == 2:
            node.mesh, node.skin = self.parse_mesh(item)

        self.node_lookup[item["name"]] = node_id
        self.gltf.nodes.append(node)
        node.children = [
            self.parse_scene_node(child) for child in item.get("children", [])
        ]
        return node_id

    def parse_mesh(self, item: dict) -> int:
        # geometry
        #    type
        #    id
        # material
        # passMaterial
        # skinning
        # modifier
        skin_id = None
        mesh_id = len(self.gltf.meshes)
        mesh = pygltflib.Mesh(
            name=item.get("name"),
            primitives=[],
        )

        geometry = item.get("geometry")
        if geometry:
            mesh.primitives = self.parse_geometry(geometry["type"], geometry["id"])
            # handle materials
            geo_mats = []
            for mat in item.get("material", []):
                geo_mats.append(self.parse_material(mat))
            if geo_mats:
                # remap material indices of primitives
                for prim in mesh.primitives:
                    prim.material = geo_mats[prim.material]

        skinning_uid = item.get("skinning")
        if skinning_uid:
            # handle skinning
            skinning = self.model["skinning"][skinning_uid]
            # matrices -> reverseBindMatrix
            matrices = flatten(skinning["matrix"])
            matrices_raw = struct.pack(f"<{len(matrices)}f", *matrices)
            bufferview_id = self.add_binary_data(
                matrices_raw, 0, f"{skinning_uid} inv bind matrices"
            )
            accessor_id = len(self.gltf.accessors)
            self.gltf.accessors.append(
                pygltflib.Accessor(
                    bufferView=bufferview_id,
                    name=f"{skinning_uid} inv bind matrices",
                    componentType=pygltflib.FLOAT,
                    count=len(matrices) // 16,
                    type="MAT4",
                )
            )

            skin = pygltflib.Skin(
                joints=skinning["bones"],
                inverseBindMatrices=accessor_id,
                name=item.get("name"),
            )
            skin_id = len(self.gltf.skins)
            self.gltf.skins.append(skin)

        self.gltf.meshes.append(mesh)
        return mesh_id, skin_id

    def parse_material(self, item) -> int:
        mat_id = len(self.gltf.materials)
        mat = pygltflib.Material(name=item.get("id"))
        mat.pbrMetallicRoughness = pygltflib.PbrMetallicRoughness()
        self.gltf.materials.append(mat)

        for uniform in item.get("uniforms", []):
            typ, key, val = uniform["type"], uniform["key"], uniform["value"]
            if not val:
                continue

            # if typ == "t":
            #    val = self.handle_texture(val)

            if key == "diffuse":
                if typ == "t":
                    val = self.handle_texture(val)
                    if val is None:
                        continue
                    mat.pbrMetallicRoughness.baseColorTexture = pygltflib.TextureInfo(
                        index=val
                    )
                else:
                    mat.pbrMetallicRoughness.baseColorFactor = val

        return mat_id

    def handle_texture(self, name) -> int:
        # config = self.scene["imgConfigData"].get(name)
        try:
            image = next(f for f in os.listdir(self.working_dir) if f.startswith(name))
        except StopIteration:
            print(f"Texture {name} not found")
            return

        image_id = len(self.gltf.images)
        fp = os.path.realpath(os.path.join(self.working_dir, image))
        raw_image = open(fp, "rb").read()
        # check image type in raw image, as the extension is not always correct
        try:
            mime_type = get_image_mimetype(raw_image)
        except Exception as e:
            print(f"Error reading image {name}: {e}")
            return

        image = pygltflib.Image(
            # uri=fp,
            mimeType=mime_type,
            name=name,
            bufferView=self.add_binary_data(open(fp, "rb").read(), 0, name),
        )
        self.gltf.images.append(image)

        texture_id = len(self.gltf.textures)
        texture = pygltflib.Texture(
            name=name,
            source=image_id,
        )
        self.gltf.textures.append(texture)
        return texture_id

    def parse_geometry(self, typ: int, g_id: str) -> List[pygltflib.Primitive]:
        geometry = self.model["geometries"].get(g_id)
        if not geometry:
            print(f"Geometry {g_id} not found")
            return

        # 1. store attributes
        attributes = pygltflib.Attributes(
            **{
                ATTRIBUTE_TO_GLTF.get(key, key): self.add_array(**val, name=key)
                for key, val in geometry["attributes"].items()
                if val is not None and not key.startswith("morphTarget")
            }
        )
        targets = []
        for i in range(256):
            target_key = f"morphTarget{i}"
            val = geometry["attributes"].get(target_key)
            if not val:
                break
            targets.append(
                pygltflib.Attributes(POSITION=self.add_array(**val, name=target_key))
            )
        # morphTargets ????

        # 2. store indices
        primitives = []
        target_count = None
        for group in geometry.get(
            "groups",
            [{"start": 0, "count": len(geometry["index"]), "materialIndex": 0}],
        ):
            start, count = group.get("start", 0), group.get("count", 0)
            indices = geometry["index"][start : start + count]
            indice_idx = self.add_array(
                itemSize=1, type=1, array=indices, name="indices"
            )
            prim = pygltflib.Primitive(
                indices=indice_idx,
                attributes=attributes,
                material=group.get("materialIndex", None),
                targets=targets,
            )
            primitives.append(prim)
            if target_count is None:
                target_count = len(targets)
            elif target_count != 0 and target_count != len(targets):
                raise NotImplementedError("Different number of targets in one geometry")

        return primitives

    def add_array(
        self,
        itemSize: int,
        type: int,
        array: memoryview,
        name: str = None,
        target: int = None,
    ) -> int:
        if array.format == "f":
            val_type = pygltflib.FLOAT
        elif array.format == "I":
            val_type = pygltflib.UNSIGNED_INT
        elif array.format == "H":
            val_type = pygltflib.UNSIGNED_SHORT
        elif array.format == "B":
            val_type = pygltflib.UNSIGNED_BYTE
        else:
            raise NotImplementedError(f"Unknown type {array.format}")

        if name.lower().startswith("uv"):
            # flip v
            array = memoryview(bytearray(array)).cast("f")
            for i in range(1, len(array), itemSize):
                array[i] = 1 - array[i]

        if target is None:
            target = (
                pygltflib.ARRAY_BUFFER
                if itemSize > 1
                else pygltflib.ELEMENT_ARRAY_BUFFER
            )

        accessor_id = len(self.gltf.accessors)
        accessor = pygltflib.Accessor(
            name=name,
            bufferView=self.add_binary_data(
                array.tobytes(),
                target=target,
                name=name,
            ),
            componentType=val_type,
            count=int(len(array) // itemSize),
            type=f"VEC{itemSize}" if itemSize > 1 else "SCALAR",
            min=[min(array[i::itemSize]) for i in range(itemSize)],
            max=[max(array[i::itemSize]) for i in range(itemSize)],
        )
        self.gltf.accessors.append(accessor)
        return accessor_id

    def add_binary_data(self, data: bytes, target: int, name: str = None) -> int:
        # Add binary data to the buffer
        # Returns the offset
        glb_data = getattr(self.gltf, "_glb_data", None)
        if glb_data is None:
            self.gltf._glb_data = b""

        buffer_view_id = len(self.gltf.bufferViews)
        buffer_view = pygltflib.BufferView(
            byteOffset=len(self.gltf._glb_data),
            byteLength=len(data),
            target=target,
            name=name,
            buffer=0,
        )
        self.gltf.bufferViews.append(buffer_view)

        pad = len(data) % 4
        if pad:
            data += b"\x00" * (4 - pad)

        self.gltf._glb_data += data

        return buffer_view_id
