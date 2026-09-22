"""Guard on the internals that carrying an example to other frames leans on.

SAM 3 reads an example box against the picture it was drawn on, so the box
cannot travel.  What travels is the prompt the geometry encoder builds from it,
handed back to the model as `text_embeds` — which works only because
`Sam3Model.forward` reads `.pooler_output` and, with no `input_boxes`, passes it
through as the whole prompt.

That is a private arrangement.  A rename is caught here, cheaply and without
weights.  The quieter danger — the halves concatenated the other way round, a
different FPN level — cannot be caught by a signature, so `encode_exemplar_prompt`
also compares its prompt against the model's own on every run.  Both matter: this
one keeps the failure legible, that one keeps it honest.

Run with:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.sam3_handler import (EXEMPLAR_THRESHOLD, ExemplarUnsupported,
                                   check_exemplar_support)


class SupportCheckTests(unittest.TestCase):
    def test_passes_on_the_installed_transformers(self) -> None:
        check_exemplar_support()          # raises if it does not

    def test_a_missing_forward_parameter_is_refused(self) -> None:
        from transformers.models.sam3 import modeling_sam3 as M

        def forward_without_text_embeds(self, pixel_values=None, input_ids=None,
                                        input_boxes=None):
            raise AssertionError("never called")

        with mock.patch.object(M.Sam3Model, "forward", forward_without_text_embeds):
            with self.assertRaises(ExemplarUnsupported) as caught:
                check_exemplar_support()
        message = str(caught.exception)
        self.assertIn("text_embeds", message)
        # The message has to say what still works, or it reads as the whole
        # feature being broken rather than one method of four.
        self.assertIn("unaffected", message)

    def test_a_geometry_encoder_that_lost_its_positions_is_refused(self) -> None:
        from transformers.models.sam3 import modeling_sam3 as M

        def geometry_without_pos(self, box_embeddings=None, box_mask=None,
                                 box_labels=None, img_feats=None):
            raise AssertionError("never called")

        with mock.patch.object(M.Sam3GeometryEncoder, "forward", geometry_without_pos):
            with self.assertRaises(ExemplarUnsupported) as caught:
                check_exemplar_support()
        self.assertIn("img_pos_embeds", str(caught.exception))

    def test_a_vision_output_without_fpn_is_refused(self) -> None:
        from transformers.models.sam3 import modeling_sam3 as M

        thinned = {k: v for k, v in M.Sam3VisionEncoderOutput.__annotations__.items()
                   if k != "fpn_position_encoding"}
        with mock.patch.object(M.Sam3VisionEncoderOutput, "__annotations__", thinned):
            with self.assertRaises(ExemplarUnsupported) as caught:
                check_exemplar_support()
        self.assertIn("fpn_position_encoding", str(caught.exception))

    def test_the_named_version_is_advisory_not_a_gate(self) -> None:
        """A patch bump that changes nothing must not brick the method."""
        import transformers

        with mock.patch.object(transformers, "__version__", "99.0.0"):
            check_exemplar_support()


class ThresholdTests(unittest.TestCase):
    def test_examples_default_above_the_text_threshold(self) -> None:
        """Swept on held-out and verified frames: a whole-object example peaked
        at 0.50-0.65, where 0.35 returned about 420 boxes for 171 objects."""
        self.assertGreaterEqual(EXEMPLAR_THRESHOLD, 0.5)


if __name__ == "__main__":
    unittest.main()
