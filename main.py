import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.inference import model_fn, predict_fn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
MODEL_BUNDLE = None

CONTENT_WEIGHTS = {
    "Video": 1.0,
    "Reading": 1.5,
    "Assignment": 3.0,
    "Quiz": 1.5,
    "Summary": 1.0,
}


class CourseRecommendationRequest(BaseModel):
    provider: str = Field(..., examples=["Coursera"])
    depth_level: str = Field(..., examples=["Beginner"])
    category: str = Field(..., examples=["Data Science: Machine Learning"])
    num_chapters: int = Field(..., ge=1, le=60)
    duration_hours: float = Field(..., gt=0)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return max(0, int(round(float(value))))
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def format_duration(total_minutes: int) -> str:
    total_minutes = max(0, int(round(total_minutes)))
    hours, minutes = divmod(total_minutes, 60)

    if hours and minutes:
        return f"{hours}h {minutes}m"
    if hours:
        return f"{hours}h"
    return f"{minutes}m"


def distribute_items(total_items: int, number_of_chapters: int) -> list[int]:
    total_items = safe_int(total_items)
    number_of_chapters = max(1, safe_int(number_of_chapters, 1))
    base_count, remainder = divmod(total_items, number_of_chapters)

    return [
        base_count + (1 if index < remainder else 0)
        for index in range(number_of_chapters)
    ]


def allocate_integer_amount(
    total: int,
    weights: list[float],
    minimum_each: int = 0,
) -> list[int]:
    total = max(0, int(round(total)))
    count = len(weights)

    if count == 0:
        return []

    cleaned_weights = [max(0.0001, safe_float(weight, 1.0)) for weight in weights]

    if minimum_each > 0 and total >= count * minimum_each:
        values = [minimum_each] * count
        remaining = total - count * minimum_each
    else:
        values = [0] * count
        remaining = total

    if remaining <= 0:
        return values

    weight_sum = sum(cleaned_weights)
    raw_values = [remaining * weight / weight_sum for weight in cleaned_weights]
    floors = [int(value) for value in raw_values]
    fractions = [
        raw_value - floor_value
        for raw_value, floor_value in zip(raw_values, floors)
    ]

    values = [value + floor_value for value, floor_value in zip(values, floors)]
    leftover = remaining - sum(floors)

    order = sorted(
        range(count),
        key=lambda index: fractions[index],
        reverse=True,
    )

    for index in order[:leftover]:
        values[index] += 1

    return values


def create_lesson(
    lesson_type: str,
    chapter_number: int,
    lesson_number: int,
    duration_minutes: int,
) -> dict[str, Any]:
    if lesson_type == "Video":
        title = f"Lesson {chapter_number}.{lesson_number}: Video Lecture"
    elif lesson_type == "Reading":
        title = f"Reading Material {chapter_number}.{lesson_number}"
    elif lesson_type == "Assignment":
        title = f"Practical Assignment {chapter_number}.{lesson_number}"
    elif lesson_type == "Quiz":
        title = f"Knowledge Check {chapter_number}.{lesson_number}"
    else:
        title = f"Chapter Summary {chapter_number}.{lesson_number}"

    return {
        "type": lesson_type,
        "title": title,
        "duration_minutes": duration_minutes,
        "duration_display": format_duration(duration_minutes),
        "is_video": lesson_type == "Video",
    }


def create_course_structure(
    prediction: dict[str, Any],
    requested_chapters: int,
    requested_duration_hours: float,
) -> dict[str, Any]:
    number_of_chapters = max(1, safe_int(requested_chapters, 1))
    total_duration_minutes = max(
        1,
        int(round(safe_float(requested_duration_hours, 1.0) * 60)),
    )

    total_videos = safe_int(prediction.get("recommended_num_videos", 0))
    total_readings = safe_int(prediction.get("recommended_num_readings", 0))
    total_assignments = safe_int(prediction.get("recommended_num_assignments", 0))
    total_quizzes = safe_int(prediction.get("recommended_num_quizzes", 0))
    total_summaries = safe_int(prediction.get("recommended_num_summaries", 0))

    total_lectures = safe_int(
        prediction.get("predicted_json_lectures", total_videos)
    )
    total_assessments = safe_int(prediction.get("predicted_json_assessments", 0))
    total_supplements = safe_int(
        prediction.get("predicted_json_supplements", total_readings)
    )

    video_distribution = distribute_items(total_videos, number_of_chapters)
    reading_distribution = distribute_items(total_readings, number_of_chapters)
    assignment_distribution = distribute_items(total_assignments, number_of_chapters)
    quiz_distribution = distribute_items(total_quizzes, number_of_chapters)
    summary_distribution = distribute_items(total_summaries, number_of_chapters)

    chapter_weights = []

    for index in range(number_of_chapters):
        weight = (
            video_distribution[index] * CONTENT_WEIGHTS["Video"]
            + reading_distribution[index] * CONTENT_WEIGHTS["Reading"]
            + assignment_distribution[index] * CONTENT_WEIGHTS["Assignment"]
            + quiz_distribution[index] * CONTENT_WEIGHTS["Quiz"]
            + summary_distribution[index] * CONTENT_WEIGHTS["Summary"]
        )
        chapter_weights.append(max(1.0, weight))

    chapter_durations = allocate_integer_amount(
        total_duration_minutes,
        chapter_weights,
        minimum_each=1,
    )

    chapters = []
    all_lessons = []

    for index in range(number_of_chapters):
        chapter_number = index + 1

        lesson_types = (
            ["Video"] * video_distribution[index]
            + ["Reading"] * reading_distribution[index]
            + ["Assignment"] * assignment_distribution[index]
            + ["Quiz"] * quiz_distribution[index]
            + ["Summary"] * summary_distribution[index]
        )

        chapter_duration = chapter_durations[index]
        lesson_weights = [
            CONTENT_WEIGHTS.get(lesson_type, 1.0)
            for lesson_type in lesson_types
        ]

        lesson_durations = allocate_integer_amount(
            chapter_duration,
            lesson_weights,
            minimum_each=1,
        )

        lessons = []

        for lesson_index, lesson_type in enumerate(lesson_types):
            lesson = create_lesson(
                lesson_type=lesson_type,
                chapter_number=chapter_number,
                lesson_number=lesson_index + 1,
                duration_minutes=lesson_durations[lesson_index],
            )
            lessons.append(lesson)
            all_lessons.append(lesson)

        def minutes_for(lesson_type: str) -> int:
            return sum(
                lesson["duration_minutes"]
                for lesson in lessons
                if lesson["type"] == lesson_type
            )

        video_minutes = minutes_for("Video")
        reading_minutes = minutes_for("Reading")
        assignment_minutes = minutes_for("Assignment")
        quiz_minutes = minutes_for("Quiz")
        summary_minutes = minutes_for("Summary")

        chapters.append(
            {
                "chapter_number": chapter_number,
                "title": f"Chapter {chapter_number}: Core Concepts",
                "videos": video_distribution[index],
                "readings": reading_distribution[index],
                "assignments": assignment_distribution[index],
                "quizzes": quiz_distribution[index],
                "summaries": summary_distribution[index],
                "duration_minutes": chapter_duration,
                "duration_display": format_duration(chapter_duration),
                "duration_percent": round(
                    chapter_duration / total_duration_minutes * 100,
                    2,
                ),
                "video_minutes": video_minutes,
                "video_duration_display": format_duration(video_minutes),
                "reading_minutes": reading_minutes,
                "reading_duration_display": format_duration(reading_minutes),
                "assignment_minutes": assignment_minutes,
                "quiz_minutes": quiz_minutes,
                "summary_minutes": summary_minutes,
                "lessons": lessons,
            }
        )

    def total_minutes_for(lesson_type: str) -> int:
        return sum(
            lesson["duration_minutes"]
            for lesson in all_lessons
            if lesson["type"] == lesson_type
        )

    total_video_minutes = total_minutes_for("Video")
    total_reading_minutes = total_minutes_for("Reading")
    total_assignment_minutes = total_minutes_for("Assignment")
    total_quiz_minutes = total_minutes_for("Quiz")
    total_summary_minutes = total_minutes_for("Summary")

    return {
        "course_title": prediction.get("input_category", "Recommended Course"),
        "provider": prediction.get("input_provider", "Course Provider"),
        "depth_level": prediction.get("input_depth_level", "Beginner"),
        "duration_hours": round(total_duration_minutes / 60, 2),
        "total_duration_minutes": sum(
            chapter["duration_minutes"] for chapter in chapters
        ),
        "total_duration_display": format_duration(total_duration_minutes),
        "number_of_chapters": number_of_chapters,
        "template": prediction.get(
            "predicted_structure_template",
            "multi_module_assessment",
        ),
        "tags": prediction.get("predicted_structure_tags", []),
        "total_videos": total_videos,
        "total_lectures": total_lectures,
        "total_readings": total_readings,
        "total_supplements": total_supplements,
        "total_assignments": total_assignments,
        "total_quizzes": total_quizzes,
        "total_summaries": total_summaries,
        "total_assessments": total_assessments,
        "total_video_minutes": total_video_minutes,
        "total_video_duration_display": format_duration(total_video_minutes),
        "total_reading_minutes": total_reading_minutes,
        "total_assignment_minutes": total_assignment_minutes,
        "total_quiz_minutes": total_quiz_minutes,
        "total_summary_minutes": total_summary_minutes,
        "chapters": chapters,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL_BUNDLE

    logger.info("Loading model from %s", MODEL_DIR)
    MODEL_BUNDLE = model_fn(str(MODEL_DIR))
    logger.info("Model loaded successfully")
    yield
    logger.info("Application shutting down")


TEMPLATES_DIR = BASE_DIR / "app" / "templates"
STATIC_DIR = BASE_DIR / "app" / "static"

app = FastAPI(
    title="Course Recommendation API",
    version="2.0.0",
    lifespan=lifespan,
)


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def root():
    return FileResponse(str(TEMPLATES_DIR / "index.html"))


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": MODEL_BUNDLE is not None,
    }


@app.post("/predict")
def predict(request: CourseRecommendationRequest):
    if MODEL_BUNDLE is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    try:
        prediction = predict_fn(
            [request.model_dump()],
            MODEL_BUNDLE,
        )[0]

        return create_course_structure(
            prediction=prediction,
            requested_chapters=request.num_chapters,
            requested_duration_hours=request.duration_hours,
        )

    except Exception as error:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error
