"""End-to-end Raster-to-Brushwork Vectorization Pipeline orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from brush_vectorizer.config import CanvasTransform, PhysicalCalibrationConfig
from brush_vectorizer.models import StrokeVectorDocument
from brush_vectorizer.pipeline.ordering import OrderingResult, run_global_path_ordering
from brush_vectorizer.pipeline.stage1_preprocess import (
    Stage1Result,
    load_luminance_image,
    run_stage1_preprocessing,
)
from brush_vectorizer.pipeline.stage2_medial_axis import (
    Stage2Result,
    run_stage2_medial_axis,
)
from brush_vectorizer.pipeline.stage3_graph import (
    Stage3Result,
    run_stage3_graph_untangling,
)
from brush_vectorizer.pipeline.stage4_classify import (
    Stage4Result,
    run_stage4_classification,
)
from brush_vectorizer.pipeline.stage5_polarity import (
    Stage5Result,
    run_stage5_polarity_resolution,
)
from brush_vectorizer.pipeline.stage6_kinematics import (
    Stage6Result,
    run_stage6_kinematics,
)


@dataclass
class PipelineExecutionArtifacts:
    """Complete pipeline execution result including intermediate stage outputs."""

    document: StrokeVectorDocument
    transform: CanvasTransform
    stage1: Stage1Result
    stage2: Stage2Result
    stage3: Stage3Result
    stage4: Stage4Result
    stage5: Stage5Result
    stage6: Stage6Result
    ordering: OrderingResult


def run_vectorization_pipeline_from_array(
    luminance: np.ndarray,
    config: PhysicalCalibrationConfig | None = None,
) -> PipelineExecutionArtifacts:
    """Run all 6 stages + global ordering on an in-memory luminance array Y in [0, 1]."""
    cfg = config or PhysicalCalibrationConfig()
    height_px, width_px = luminance.shape[:2]
    transform = CanvasTransform.from_image_and_canvas(
        width_px=width_px,
        height_px=height_px,
        canvas_dims_mm=cfg.canvas_dims_mm,
    )

    # Stage 1: Preprocessing & Tonal Segmentation
    s1 = run_stage1_preprocessing(luminance, cfg)

    # Stage 2: Euclidean Distance Transform & Medial Axis Extraction
    s2 = run_stage2_medial_axis(s1.stroke_mask, cfg)

    # Stage 3: Graph Construction & Momentum Junction Untangling
    s3 = run_stage3_graph_untangling(
        pruned_skeleton=s2.pruned_skeleton,
        edt_radius_px=s2.edt_radius_px,
        density_map=s1.density_map,
        config=cfg,
    )

    # Stage 4: Stroke Classification (Calligraphic vs. Broad Wash)
    s4 = run_stage4_classification(
        untangled_paths=s3.untangled_paths,
        stroke_mask=s1.stroke_mask,
        edt_radius_px=s2.edt_radius_px,
        density_map=s1.density_map,
        transform=transform,
        config=cfg,
    )

    # Stage 5: Polarity Resolution (Start vs. End Detection)
    s5 = run_stage5_polarity_resolution(
        classified_paths=s4.classified_paths,
        density_map=s1.density_map,
        config=cfg,
    )

    # Stage 6: Parametric Spline Fitting & Kinematics Compensation
    s6 = run_stage6_kinematics(
        oriented_paths=s5.oriented_paths,
        transform=transform,
        config=cfg,
    )

    # Section 4: Global Path Ordering (Plotter Optimization)
    ordering = run_global_path_ordering(
        debug_data=s6.debug_data,
        config=cfg,
    )

    document = StrokeVectorDocument(
        canvas_dims_mm=cfg.canvas_dims_mm,
        source_dims_px=(width_px, height_px),
        stroke_count=len(ordering.ordered_strokes),
        strokes=ordering.ordered_strokes,
    )

    return PipelineExecutionArtifacts(
        document=document,
        transform=transform,
        stage1=s1,
        stage2=s2,
        stage3=s3,
        stage4=s4,
        stage5=s5,
        stage6=s6,
        ordering=ordering,
    )


def run_vectorization_pipeline(
    image_path: str | Path,
    config: PhysicalCalibrationConfig | None = None,
) -> PipelineExecutionArtifacts:
    """Load an image from disk and execute the full vectorization pipeline."""
    luminance = load_luminance_image(image_path)
    return run_vectorization_pipeline_from_array(
        luminance=luminance,
        config=config,
    )
