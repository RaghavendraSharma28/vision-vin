from mulkiya_vin.ocr import OcrEngine
from mulkiya_vin.pipeline import MulkiyaVinPipeline
from mulkiya_vin.types import OcrResult, VehicleMetadata
from mulkiya_vin.vlm import VlmExtractor


class FakeOcr(OcrEngine):
    def extract(self, image_path):
        return [OcrResult("MNTBAABI7L0001234", 0.94)]


class FakeVlm(VlmExtractor):
    def extract(self, image_path):
        return VehicleMetadata(make="Nissan", year="2020")


def test_pipeline_selects_corrected_vin(tmp_path):
    image_path = tmp_path / "mulkiya.jpg"
    image_path.write_bytes(b"not actually read")
    pipeline = MulkiyaVinPipeline(ocr_engine=FakeOcr(), vlm_extractor=FakeVlm())

    result = pipeline.process(image_path, request_id="req-1")

    assert result.final_vin == "MNTBAAB17L0001234"
    assert not result.human_review_required
