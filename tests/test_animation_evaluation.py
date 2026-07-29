"""Tests for animation evaluation and baking (ufbx 0.23 APIs).

Uses the skinned `maya_game_sausage` wiggle animation from the upstream ufbx
test data (3 joints, ~0.8s clip).
"""

import os

import numpy as np
import pytest

import ufbx

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "maya_game_sausage_7500_binary_wiggle.fbx")

pytestmark = pytest.mark.skipif(not os.path.exists(FIXTURE), reason="animation fixture missing")


@pytest.fixture(scope="module")
def scene():
    scene = ufbx.load_file(FIXTURE, ignore_geometry=True)
    yield scene
    scene.close()


class TestAnimOnlyLoading:
    def test_ignore_geometry_load(self, scene):
        """Scene loads with geometry skipped; nodes and stacks remain."""
        names = [node.name for node in scene.nodes]
        assert "joint1" in names
        assert "joint2" in names
        assert "joint3" in names

    def test_memory_load_matches_file_load(self, scene):
        """Buffer loading preserves animation hierarchy and filtering options."""
        with open(FIXTURE, "rb") as fixture_file:
            data = fixture_file.read()
        with ufbx.load_memory(data, ignore_geometry=True, ignore_embedded=True) as memory_scene:
            assert [node.name for node in memory_scene.nodes] == [node.name for node in scene.nodes]
            assert len(memory_scene.meshes) == len(scene.meshes)
            assert all(mesh.num_vertices == 0 for mesh in memory_scene.meshes)
            assert len(memory_scene.anim_stacks) == len(scene.anim_stacks)

    def test_anim_stack_exposes_anim(self, scene):
        stacks = scene.anim_stacks
        assert len(stacks) >= 1
        assert stacks[0].anim is not None

    def test_scene_default_anim(self, scene):
        assert scene.anim is not None

    def test_node_typed_id_roundtrip(self, scene):
        for i, node in enumerate(scene.nodes):
            assert node.typed_id == i


class TestEvaluateTransform:
    def test_evaluate_returns_transform(self, scene):
        joint = next(node for node in scene.nodes if node.name == "joint2")
        transform = joint.evaluate_transform(0.5)
        assert isinstance(transform, ufbx.Transform)
        # Rotation should be a valid unit quaternion.
        q = transform.rotation
        norm = (q.x**2 + q.y**2 + q.z**2 + q.w**2) ** 0.5
        assert norm == pytest.approx(1.0, abs=1e-6)

    def test_evaluate_changes_over_time(self, scene):
        joint = next(node for node in scene.nodes if node.name == "joint2")
        q_start = joint.evaluate_transform(0.1).rotation
        q_end = joint.evaluate_transform(0.7).rotation
        assert abs(q_start.z - q_end.z) > 1e-3

    def test_evaluate_with_explicit_anim(self, scene):
        anim = scene.anim_stacks[0].anim
        joint = next(node for node in scene.nodes if node.name == "joint1")
        transform = joint.evaluate_transform(0.25, anim=anim)
        assert isinstance(transform, ufbx.Transform)


class TestCurveEvaluate:
    def test_curve_evaluate_within_range(self, scene):
        curves = scene.anim_curves
        assert len(curves) > 0
        curve = max(curves, key=lambda c: c.num_keyframes)
        value_min = curve.evaluate(curve.min_time)
        value_max = curve.evaluate(curve.max_time)
        assert curve.min_value - 1e-6 <= value_min <= curve.max_value + 1e-6
        assert curve.min_value - 1e-6 <= value_max <= curve.max_value + 1e-6

    def test_curve_evaluate_default_value(self, scene):
        curve = scene.anim_curves[0]
        assert isinstance(curve.evaluate(0.0, 0.0), float)


class TestBakeAnim:
    def test_bake_produces_nodes(self, scene):
        baked = scene.bake_anim(resample_rate=30.0, trim_start_time=True)
        assert isinstance(baked, ufbx.BakedAnim)
        assert len(baked.nodes) == 3  # three animated joints
        assert baked.playback_duration > 0.5

    def test_baked_keys_shapes_and_times(self, scene):
        baked = scene.bake_anim(resample_rate=30.0, trim_start_time=True)
        for baked_node in baked.nodes:
            num_r = len(baked_node.rotation_times)
            assert baked_node.rotation_values.shape == (num_r, 4)
            assert baked_node.translation_values.shape == (len(baked_node.translation_times), 3)
            assert baked_node.scale_values.shape == (len(baked_node.scale_times), 3)
            # Times must start at ~0 (trimmed) and be strictly increasing.
            assert baked_node.rotation_times[0] == pytest.approx(0.0, abs=1e-9)
            assert np.all(np.diff(baked_node.rotation_times) > 0)

    def test_baked_rotations_are_unit_quaternions(self, scene):
        baked = scene.bake_anim(resample_rate=30.0)
        for baked_node in baked.nodes:
            norms = np.linalg.norm(baked_node.rotation_values, axis=1)
            np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_baked_node_maps_to_scene_node(self, scene):
        baked = scene.bake_anim()
        names = {scene.nodes[baked_node.typed_id].name for baked_node in baked.nodes}
        assert names == {"joint1", "joint2", "joint3"}

    def test_bake_matches_evaluate_transform(self, scene):
        """Baked keys must agree with direct per-time evaluation."""
        baked = scene.bake_anim(resample_rate=30.0)
        baked_node = next(b for b in baked.nodes if scene.nodes[b.typed_id].name == "joint2")
        node = scene.nodes[baked_node.typed_id]
        for key_index in (0, len(baked_node.rotation_times) // 2, len(baked_node.rotation_times) - 1):
            t = baked_node.rotation_times[key_index]
            q_eval = node.evaluate_transform(float(t)).rotation
            q_baked = baked_node.rotation_values[key_index]
            dot = abs(
                q_baked[0] * q_eval.x + q_baked[1] * q_eval.y + q_baked[2] * q_eval.z + q_baked[3] * q_eval.w
            )
            assert dot == pytest.approx(1.0, abs=1e-6)

    def test_baked_arrays_survive_scene_close(self):
        scene = ufbx.load_file(FIXTURE, ignore_geometry=True)
        baked = scene.bake_anim(resample_rate=30.0)
        rotations = baked.nodes[0].rotation_values
        scene.close()
        # Arrays were copied out; must remain readable after close.
        assert np.isfinite(rotations).all()
