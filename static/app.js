const form = document.getElementById('course-form');
const submitButton = document.getElementById('submit-button');
const emptyState = document.getElementById('empty-state');
const resultContent = document.getElementById('result-content');
const messageArea = document.getElementById('message-area');

const escapeHtml = (value) => String(value ?? '')
  .replaceAll('&', '&amp;')
  .replaceAll('<', '&lt;')
  .replaceAll('>', '&gt;')
  .replaceAll('"', '&quot;')
  .replaceAll("'", '&#039;');

const iconFor = (type) => ({
  Video: '▶',
  Reading: '▤',
  Assignment: '✓',
  Quiz: '?',
  Summary: '≡'
}[type] || '•');

const stat = (label, value) => `
  <div class="stat"><div class="stat-label">${escapeHtml(label)}</div><div class="stat-value">${escapeHtml(value)}</div></div>`;

function renderChapter(chapter, index) {
  const lessons = (chapter.lessons || []).map((lesson) => `
    <div class="lesson">
      <div class="lesson-icon">${iconFor(lesson.type)}</div>
      <div class="lesson-main">
        <div class="lesson-type">${escapeHtml(lesson.type)}</div>
        <div class="lesson-title">${escapeHtml(lesson.title)}</div>
      </div>
      <div class="lesson-duration">${escapeHtml(lesson.duration_display)}</div>
    </div>`).join('');

  return `
    <details class="chapter" ${index === 0 ? 'open' : ''}>
      <summary class="chapter-summary">
        <div class="chapter-summary-left">
          <div class="chapter-number">${String(chapter.chapter_number).padStart(2, '0')}</div>
          <div class="chapter-name">
            <h4>${escapeHtml(chapter.title)}</h4>
            <p>${chapter.videos} videos · ${chapter.readings} readings · ${chapter.assignments} assignments</p>
          </div>
        </div>
        <div class="chapter-time"><strong>${escapeHtml(chapter.duration_display)}</strong><span>${escapeHtml(chapter.duration_percent)}% of course</span></div>
      </summary>
      <div class="chapter-body">
        <div class="chapter-progress" aria-label="Chapter duration percentage">
          <progress class="chapter-progress-fill" value="${Number(chapter.duration_percent) || 0}" max="100"></progress>
        </div>
        <div class="chapter-stats">
          <span class="chapter-stat">Videos: ${chapter.videos}</span>
          <span class="chapter-stat">Readings: ${chapter.readings}</span>
          <span class="chapter-stat">Assignments: ${chapter.assignments}</span>
          <span class="chapter-stat">Quizzes: ${chapter.quizzes}</span>
          <span class="chapter-stat">Summaries: ${chapter.summaries}</span>
          <span class="chapter-stat">Video time: ${escapeHtml(chapter.video_duration_display)}</span>
        </div>
        <div class="lesson-list">${lessons || '<p class="helper-text">No individual lessons were generated for this chapter.</p>'}</div>
      </div>
    </details>`;
}

function renderCourse(course) {
  const chapters = (course.chapters || []).map(renderChapter).join('');
  const tags = (course.tags || []).map(tag => `<span class="tag">${escapeHtml(tag)}</span>`).join('') || '<span class="tag">No tags predicted</span>';

  resultContent.innerHTML = `
    <div class="result-header">
      <div><p class="eyebrow">Generated Learning Plan</p><h2>${escapeHtml(course.course_title)}</h2>
      <p class="result-meta">${escapeHtml(course.provider)} · ${escapeHtml(course.depth_level)} · ${course.number_of_chapters} chapters</p></div>
      <span class="template-badge">${escapeHtml(course.template)}</span>
    </div>
    <div class="duration-hero">
      <div class="duration-card primary"><div class="duration-label">Total Learning Duration</div><div class="duration-value">${escapeHtml(course.total_duration_display)}</div><div class="duration-caption">Complete course estimate</div></div>
      <div class="duration-card"><div class="duration-label">Video Learning Time</div><div class="duration-value">${escapeHtml(course.total_video_duration_display)}</div><div class="duration-caption">Across all videos</div></div>
      <div class="duration-card"><div class="duration-label">Chapter Count</div><div class="duration-value">${course.number_of_chapters}</div><div class="duration-caption">Planned learning modules</div></div>
    </div>
    <div class="section-heading"><h3>Content Overview</h3><span>Model predictions</span></div>
    <div class="stats-grid">
      ${stat('Chapters', course.number_of_chapters)}${stat('Videos', course.total_videos)}${stat('Readings', course.total_readings)}${stat('Assignments', course.total_assignments)}${stat('Quizzes', course.total_quizzes)}${stat('Summaries', course.total_summaries)}${stat('Assessments', course.total_assessments)}${stat('Lectures', course.total_lectures)}
    </div>
    <div class="section-heading"><h3>Time Allocation</h3><span>Estimated minutes</span></div>
    <div class="breakdown">
      <div class="breakdown-item"><strong>Videos</strong><span>${escapeHtml(course.total_video_duration_display)}</span></div>
      <div class="breakdown-item"><strong>Readings</strong><span>${course.total_reading_minutes}m</span></div>
      <div class="breakdown-item"><strong>Assignments</strong><span>${course.total_assignment_minutes}m</span></div>
      <div class="breakdown-item"><strong>Quizzes</strong><span>${course.total_quiz_minutes}m</span></div>
      <div class="breakdown-item"><strong>Summaries</strong><span>${course.total_summary_minutes}m</span></div>
    </div>
    <div class="section-heading"><h3>Recommended Course Structure</h3><span>Open a chapter to view lessons</span></div>
    ${chapters}
    <div class="section-heading"><h3>Structure Tags</h3></div><div class="tag-list">${tags}</div>
    <details class="json-preview"><summary>View Generated Structure JSON</summary><pre>${escapeHtml(JSON.stringify(course, null, 2))}</pre></details>`;

  emptyState.hidden = true;
  resultContent.hidden = false;
}

function showError(message) {
  messageArea.innerHTML = `<div class="error"><strong>Prediction Error</strong><br>${escapeHtml(message)}</div>`;
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  messageArea.innerHTML = '';
  submitButton.disabled = true;
  submitButton.textContent = 'Generating…';

  const payload = {
    provider: document.getElementById('provider').value.trim(),
    depth_level: document.getElementById('depth_level').value,
    category: document.getElementById('category').value.trim(),
    num_chapters: Number(document.getElementById('num_chapters').value),
    duration_hours: Number(document.getElementById('duration_hours').value)
  };

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'The server could not generate the course.');
    renderCourse(data);
  } catch (error) {
    showError(error.message);
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = 'Generate Course Structure';
  }
});
