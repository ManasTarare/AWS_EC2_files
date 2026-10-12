
"""
Local and SageMaker inference handler for:

all_course_recommendation_models_multilabel.joblib

Expected model directory:

model/
    all_course_recommendation_models_multilabel.joblib

The file supports:

    model_fn()
    input_fn()
    predict_fn()
    output_fn()

The model predicts numerical course attributes and structure tags.
A deterministic structure generator converts those predictions into
a complete course outline.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd


MODEL_FILENAME = (
    "all_course_recommendation_models_multilabel.joblib"
)


FALLBACK_NUM_CHAPTERS_MAX = 60


# ==========================================================================
# 1. LOAD MODEL
# ==========================================================================

def model_fn(model_dir):
    """
    Load the saved joblib model bundle.
    """

    model_path = os.path.join(
        model_dir,
        MODEL_FILENAME,
    )

    if not os.path.exists(model_path):

        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    bundle = joblib.load(model_path)

    required_keys = [
        "trained_regressors",
        "trained_multilabel_classifiers",
        "best_regressor",
        "best_multilabel_classifier",
        "best_regression_model_name",
        "best_multilabel_model_name",
        "target_columns",
        "feature_columns",
        "provider_to_id",
        "category_to_id",
        "normalization_meta",
        "mlb",
    ]

    missing = [
        key
        for key in required_keys
        if key not in bundle
    ]

    if missing:

        raise ValueError(
            "Model bundle is missing expected keys: "
            f"{missing}"
        )

    return bundle


# ==========================================================================
# 2. INPUT PROCESSING
# ==========================================================================

def input_fn(
    request_body,
    content_type="application/json",
):
    """
    Parse JSON input.

    Supported formats:

    Single object:
    {
        "provider": "Coursera",
        "depth_level": "Beginner",
        "category": "Data Science: Machine Learning",
        "num_chapters": 5,
        "duration_hours": 8.0
    }

    Multiple objects:
    {
        "instances": [
            {...},
            {...}
        ]
    }

    Or a JSON list:
    [
        {...},
        {...}
    ]
    """

    if content_type != "application/json":

        raise ValueError(
            f"Unsupported content type: {content_type}"
        )

    if isinstance(request_body, bytes):

        request_body = request_body.decode(
            "utf-8"
        )

    payload = json.loads(request_body)

    if (
        isinstance(payload, dict)
        and "instances" in payload
    ):

        records = payload["instances"]

    elif isinstance(payload, list):

        records = payload

    else:

        records = [payload]

    if not isinstance(records, list):

        raise ValueError(
            "Input records must be a list."
        )

    return records


# ==========================================================================
# 3. FEATURE ENGINEERING HELPERS
# ==========================================================================

def _normalize_count(
    value,
    original_col,
    normalization_meta,
):
    """
    Normalize count features using the metadata
    saved in the model bundle.
    """

    count_max = normalization_meta.get(
        "count_max",
        {},
    )

    max_value = count_max.get(
        original_col
    )

    if max_value is None:

        max_value = FALLBACK_NUM_CHAPTERS_MAX

    max_value = float(max_value)

    if max_value <= 0:

        return 0.0

    return float(value) / max_value


def _normalize_duration(
    hours,
    normalization_meta,
):
    """
    Normalize duration using saved metadata.
    """

    hours = max(
        0.1,
        float(hours),
    )

    duration_min = float(
        normalization_meta.get(
            "duration_min",
            0.0,
        )
    )

    duration_max = float(
        normalization_meta.get(
            "duration_max",
            1.0,
        )
    )

    denominator = (
        duration_max - duration_min
    )

    if denominator <= 0:

        return 0.0

    return (
        hours - duration_min
    ) / denominator


def _encode_depth(depth_level):
    """
    Encode course difficulty.
    """

    if isinstance(
        depth_level,
        (int, np.integer),
    ):

        return int(depth_level)

    mapping = {
        "beginner": 0,
        "intermediate": 1,
        "advanced": 2,
        "not specified": -1,
        "unspecified": -1,
        "unknown": -1,
    }

    return mapping.get(
        str(depth_level)
        .strip()
        .lower(),
        -1,
    )


def _encode_provider(
    provider,
    provider_to_id,
):
    """
    Encode provider using the saved mapping.
    """

    if isinstance(
        provider,
        (int, np.integer),
    ):

        return int(provider)

    provider = str(provider).strip()

    if provider in provider_to_id:

        return int(
            provider_to_id[provider]
        )

    title_provider = provider.title()

    if title_provider in provider_to_id:

        return int(
            provider_to_id[title_provider]
        )

    return 1


def _encode_category_text(
    category_text,
    category_to_id,
):
    """
    Convert category text into the format
    used by the training pipeline.
    """

    raw = str(category_text).strip()

    if (
        raw.replace(",", "")
        .replace(" ", "")
        .isdigit()
    ):

        return raw.replace(" ", "")

    parts = [
        part.strip()
        for part in raw.split(";")
        if part.strip()
    ]

    ids = []

    for part in parts:

        if part in category_to_id:

            ids.append(
                str(category_to_id[part])
            )

            continue

        matches = [
            category_id
            for name, category_id
            in category_to_id.items()
            if name.lower() == part.lower()
        ]

        if matches:

            ids.append(
                str(matches[0])
            )

    if ids:

        return ",".join(ids)

    return raw


def _make_input_row(
    provider,
    depth_level,
    category,
    num_chapters,
    duration_hours,
    bundle,
):
    """
    Build the exact feature row required by
    the trained regression and classification models.
    """

    provider_to_id = bundle[
        "provider_to_id"
    ]

    category_to_id = bundle[
        "category_to_id"
    ]

    normalization_meta = bundle[
        "normalization_meta"
    ]

    num_chapters = max(
        1,
        int(round(float(num_chapters))),
    )

    duration_hours = max(
        0.1,
        float(duration_hours),
    )

    return pd.DataFrame(
        [
            {
                "provider_encoded": (
                    _encode_provider(
                        provider,
                        provider_to_id,
                    )
                ),

                "depth_level": (
                    _encode_depth(
                        depth_level
                    )
                ),

                "category": (
                    _encode_category_text(
                        category,
                        category_to_id,
                    )
                ),

                "num_chapters_norm": (
                    _normalize_count(
                        num_chapters,
                        "num_chapters",
                        normalization_meta,
                    )
                ),

                "duration_hours_norm": (
                    _normalize_duration(
                        duration_hours,
                        normalization_meta,
                    )
                ),
            }
        ]
    )


def _round_count(value):
    """
    Convert a prediction into a non-negative integer.
    """

    return int(
        max(
            0,
            round(float(value)),
        )
    )


# ==========================================================================
# 4. STRUCTURE CLASSIFICATION HELPERS
# ==========================================================================

def _tags_to_template(tags):
    """
    Convert multilabel tags into a structure template name.
    """

    tags = set(tags)

    is_multi = (
        "multi_module" in tags
    )

    is_single = (
        "single_module" in tags
    )

    has_assessment = (
        "has_assessment" in tags
    )

    has_summary = (
        "has_summary" in tags
    )

    if (
        "lecture_reading_only" in tags
        and not has_assessment
    ):

        return "lecture_reading_only"

    if (
        is_multi
        and has_assessment
        and has_summary
    ):

        return "multi_module_assessment_summary"

    if (
        is_multi
        and has_assessment
    ):

        return "multi_module_assessment"

    if (
        is_single
        and has_assessment
        and has_summary
    ):

        return "single_module_assessment_summary"

    if (
        is_single
        and has_assessment
    ):

        return "single_module_assessment"

    if "simple_content" in tags:

        return "simple_content"

    return "multi_module_assessment"


# ==========================================================================
# 5. STRUCTURE GENERATION HELPERS
# ==========================================================================

def _distribute_items(
    total_items,
    num_chapters,
):
    """
    Distribute items across chapters as evenly
    as possible.

    Example:

    7 items across 3 chapters:

    [3, 2, 2]
    """

    total_items = max(
        0,
        int(total_items),
    )

    num_chapters = max(
        1,
        int(num_chapters),
    )

    base_count = (
        total_items // num_chapters
    )

    remainder = (
        total_items % num_chapters
    )

    distribution = []

    for index in range(num_chapters):

        count = base_count

        if index < remainder:

            count += 1

        distribution.append(count)

    return distribution


def _create_course_structure(
    category,
    duration_hours,
    num_chapters,
    num_videos,
    num_readings,
    num_assignments,
    num_quizzes,
    num_summaries,
    predicted_template,
    predicted_tags,
):
    """
    Generate a structured course outline from
    the model's numerical predictions.

    Important:

    The ML model predicts counts and tags.
    This function creates a deterministic outline
    using those predictions.

    It does not generate subject-specific lesson
    titles using a language model.
    """

    num_chapters = max(
        1,
        int(num_chapters),
    )

    video_distribution = _distribute_items(
        num_videos,
        num_chapters,
    )

    reading_distribution = _distribute_items(
        num_readings,
        num_chapters,
    )

    assignment_distribution = _distribute_items(
        num_assignments,
        num_chapters,
    )

    quiz_distribution = _distribute_items(
        num_quizzes,
        num_chapters,
    )

    summary_distribution = _distribute_items(
        num_summaries,
        num_chapters,
    )

    chapters = []

    for index in range(num_chapters):

        chapter_number = index + 1

        if chapter_number == 1:

            chapter_title = (
                "Introduction and Foundations"
            )

        elif chapter_number == num_chapters:

            chapter_title = (
                "Final Application and Course Wrap Up"
            )

        else:

            chapter_title = (
                f"Core Topic {chapter_number}"
            )

        chapter = {
            "chapter_number": chapter_number,
            "title": chapter_title,
            "videos": [],
            "readings": [],
            "assignments": [],
            "quizzes": [],
            "summaries": [],
        }

        # --------------------------------------------------------------
        # Videos
        # --------------------------------------------------------------

        for video_number in range(
            1,
            video_distribution[index] + 1,
        ):

            chapter["videos"].append(
                {
                    "video_number": video_number,
                    "title": (
                        "Concept explanation and examples"
                    ),
                    "type": "video",
                }
            )

        # --------------------------------------------------------------
        # Readings
        # --------------------------------------------------------------

        for reading_number in range(
            1,
            reading_distribution[index] + 1,
        ):

            chapter["readings"].append(
                {
                    "reading_number": reading_number,
                    "title": (
                        "Supporting notes and references"
                    ),
                    "type": "reading",
                }
            )

        # --------------------------------------------------------------
        # Assignments
        # --------------------------------------------------------------

        for assignment_number in range(
            1,
            assignment_distribution[index] + 1,
        ):

            chapter["assignments"].append(
                {
                    "assignment_number": (
                        assignment_number
                    ),
                    "title": (
                        "Practice task or project activity"
                    ),
                    "type": "assessment",
                }
            )

        # --------------------------------------------------------------
        # Quizzes
        # --------------------------------------------------------------

        for quiz_number in range(
            1,
            quiz_distribution[index] + 1,
        ):

            chapter["quizzes"].append(
                {
                    "quiz_number": quiz_number,
                    "title": (
                        "Knowledge check quiz"
                    ),
                    "type": "quiz",
                }
            )

        # --------------------------------------------------------------
        # Summaries
        # --------------------------------------------------------------

        for summary_number in range(
            1,
            summary_distribution[index] + 1,
        ):

            chapter["summaries"].append(
                {
                    "summary_number": (
                        summary_number
                    ),
                    "title": (
                        "Chapter summary"
                    ),
                    "type": "summary",
                }
            )

        chapters.append(chapter)

    structure_json = {
        "course_category": category,
        "estimated_duration_hours": float(
            duration_hours
        ),
        "total_chapters": num_chapters,
        "total_videos": int(num_videos),
        "total_readings": int(num_readings),
        "total_assignments": int(num_assignments),
        "total_quizzes": int(num_quizzes),
        "total_summaries": int(num_summaries),
        "structure_template": predicted_template,
        "structure_tags": predicted_tags,
        "chapters": chapters,
    }

    text_lines = []

    text_lines.append(
        "Recommended course structure"
    )

    text_lines.append(
        f"Course category: {category}"
    )

    text_lines.append(
        f"Estimated duration: {duration_hours} hours"
    )

    text_lines.append(
        f"Total chapters: {num_chapters}"
    )

    text_lines.append(
        f"Total videos: {num_videos}"
    )

    text_lines.append(
        f"Total readings: {num_readings}"
    )

    text_lines.append(
        f"Total assignments: {num_assignments}"
    )

    text_lines.append(
        f"Total quizzes: {num_quizzes}"
    )

    text_lines.append(
        f"Total summaries: {num_summaries}"
    )

    text_lines.append("")

    for chapter in chapters:

        text_lines.append(
            f"Chapter {chapter['chapter_number']}: "
            f"{chapter['title']}"
        )

        for video in chapter["videos"]:

            text_lines.append(
                f"  - Video "
                f"{video['video_number']}: "
                f"{video['title']}"
            )

        for reading in chapter["readings"]:

            text_lines.append(
                f"  - Reading "
                f"{reading['reading_number']}: "
                f"{reading['title']}"
            )

        for assignment in chapter["assignments"]:

            text_lines.append(
                f"  - Assignment "
                f"{assignment['assignment_number']}: "
                f"{assignment['title']}"
            )

        for quiz in chapter["quizzes"]:

            text_lines.append(
                f"  - Quiz "
                f"{quiz['quiz_number']}: "
                f"{quiz['title']}"
            )

        for summary in chapter["summaries"]:

            text_lines.append(
                f"  - Summary "
                f"{summary['summary_number']}: "
                f"{summary['title']}"
            )

        text_lines.append("")

    return {
        "recommended_structure": (
            "\n".join(text_lines)
        ),
        "recommended_structure_json": (
            structure_json
        ),
    }


# ==========================================================================
# 6. PREDICTION
# ==========================================================================

def _predict_one(
    record,
    bundle,
):
    """
    Generate one complete course recommendation.
    """

    provider = record.get(
        "provider"
    )

    depth_level = record.get(
        "depth_level",
        "Beginner",
    )

    category = record.get(
        "category"
    )

    num_chapters = record.get(
        "num_chapters",
        4,
    )

    duration_hours = record.get(
        "duration_hours",
        6.0,
    )

    model_name = record.get(
        "model_name"
    )

    if provider is None:

        raise ValueError(
            "The 'provider' field is required."
        )

    if category is None:

        raise ValueError(
            "The 'category' field is required."
        )

    num_chapters = max(
        1,
        int(round(float(num_chapters))),
    )

    duration_hours = max(
        0.1,
        float(duration_hours),
    )

    trained_regressors = bundle[
        "trained_regressors"
    ]

    target_columns = bundle[
        "target_columns"
    ]

    mlb = bundle[
        "mlb"
    ]

    # --------------------------------------------------------------
    # Select regression model
    # --------------------------------------------------------------

    if (
        model_name is None
        or model_name == "Best available"
    ):

        regressor = bundle[
            "best_regressor"
        ]

        regression_model_name = bundle[
            "best_regression_model_name"
        ]

    else:

        if model_name not in trained_regressors:

            raise ValueError(
                f"Unknown model_name: {model_name}. "
                f"Available models: "
                f"{list(trained_regressors.keys())}"
            )

        regressor = trained_regressors[
            model_name
        ]

        regression_model_name = model_name

    # --------------------------------------------------------------
    # Prepare model input
    # --------------------------------------------------------------

    input_row = _make_input_row(
        provider=provider,
        depth_level=depth_level,
        category=category,
        num_chapters=num_chapters,
        duration_hours=duration_hours,
        bundle=bundle,
    )

    # --------------------------------------------------------------
    # Regression prediction
    # --------------------------------------------------------------

    # The training pipeline uses log1p(target).
    # Reverse that transformation using expm1.

    pred_log = regressor.predict(
        input_row
    )[0]

    pred_real = np.expm1(
        pred_log
    )

    pred_real = np.clip(
        pred_real,
        0,
        None,
    )

    pred_counts = dict(
        zip(
            target_columns,
            pred_real,
        )
    )

    # --------------------------------------------------------------
    # Multilabel classification prediction
    # --------------------------------------------------------------

    multilabel_classifier = bundle[
        "best_multilabel_classifier"
    ]

    tag_binary = multilabel_classifier.predict(
        input_row
    )[0]

    classes = list(
        getattr(
            mlb,
            "classes_",
            [],
        )
    )

    if len(classes) != len(tag_binary):

        classes = [
            f"tag_{index}"
            for index in range(
                len(tag_binary)
            )
        ]

    predicted_tags = [
        tag
        for tag, value in zip(
            classes,
            tag_binary,
        )
        if value == 1
    ]

    predicted_template_label = (
        _tags_to_template(
            predicted_tags
        )
    )

    # --------------------------------------------------------------
    # Extract numerical predictions
    # --------------------------------------------------------------

    recommended_num_videos = _round_count(
        pred_counts["orig_num_videos"]
    )

    recommended_num_readings = _round_count(
        pred_counts["orig_num_readings"]
    )

    recommended_num_assignments = _round_count(
        pred_counts["orig_num_assignments"]
    )

    recommended_num_quizzes = _round_count(
        pred_counts["orig_num_quizzes"]
    )

    recommended_num_summaries = _round_count(
        pred_counts["orig_summary"]
    )

    predicted_json_modules = _round_count(
        pred_counts["json_num_modules"]
    )

    predicted_json_lectures = _round_count(
        pred_counts["json_num_lectures"]
    )

    predicted_json_assessments = _round_count(
        pred_counts["json_num_assessments"]
    )

    predicted_json_supplements = _round_count(
        pred_counts["json_num_supplements"]
    )

    # --------------------------------------------------------------
    # Generate complete structure
    # --------------------------------------------------------------

    generated_structure = (
        _create_course_structure(
            category=category,
            duration_hours=duration_hours,
            num_chapters=num_chapters,
            num_videos=recommended_num_videos,
            num_readings=recommended_num_readings,
            num_assignments=recommended_num_assignments,
            num_quizzes=recommended_num_quizzes,
            num_summaries=recommended_num_summaries,
            predicted_template=(
                predicted_template_label
            ),
            predicted_tags=predicted_tags,
        )
    )

    # --------------------------------------------------------------
    # Final response
    # --------------------------------------------------------------

    result = {
        "regression_model": (
            regression_model_name
        ),

        "structure_multilabel_model": (
            bundle[
                "best_multilabel_model_name"
            ]
        ),

        "input_provider": provider,

        "input_depth_level": depth_level,

        "input_category": category,

        "recommended_num_chapters": (
            num_chapters
        ),

        "recommended_duration_hours": (
            duration_hours
        ),

        "recommended_num_videos": (
            recommended_num_videos
        ),

        "recommended_num_readings": (
            recommended_num_readings
        ),

        "recommended_num_assignments": (
            recommended_num_assignments
        ),

        "recommended_num_quizzes": (
            recommended_num_quizzes
        ),

        "recommended_num_summaries": (
            recommended_num_summaries
        ),

        "predicted_json_modules": (
            predicted_json_modules
        ),

        "predicted_json_lectures": (
            predicted_json_lectures
        ),

        "predicted_json_assessments": (
            predicted_json_assessments
        ),

        "predicted_json_supplements": (
            predicted_json_supplements
        ),

        "predicted_structure_template": (
            predicted_template_label
        ),

        "predicted_structure_tags": (
            predicted_tags
        ),

        "recommended_structure": (
            generated_structure[
                "recommended_structure"
            ]
        ),

        "recommended_structure_json": (
            generated_structure[
                "recommended_structure_json"
            ]
        ),
    }

    return result


def predict_fn(
    records,
    bundle,
):
    """
    Predict all input records.
    """

    return [
        _predict_one(
            record,
            bundle,
        )
        for record in records
    ]


# ==========================================================================
# 7. OUTPUT PROCESSING
# ==========================================================================

def output_fn(
    predictions,
    accept="application/json",
):
    """
    Return JSON response.
    """

    if accept == "application/json":

        body = (
            predictions[0]
            if len(predictions) == 1
            else predictions
        )

        return (
            json.dumps(
                body,
                indent=2,
            ),
            accept,
        )

    raise ValueError(
        f"Unsupported accept type: {accept}"
    )