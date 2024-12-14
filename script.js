// script.js
let quizQuestions = [] // Load this dynamically from your JSON file
let quizAnswers = [] // Load this dynamically from your JSON file
let currentQuestions = []
let showAnswers = false
let quizStarted = false

// Utility Functions
function shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1))
        ;[array[i], array[j]] = [array[j], array[i]]
    }
}

function extractTags(questions) {
    const tags = new Set()
    questions.forEach(q => q.tags.forEach(tag => tags.add(tag)))
    return Array.from(tags).sort()
}

function renderFilters(tags) {
    const tagFiltersDiv = document.getElementById("tag-filters")
    tagFiltersDiv.innerHTML = ""
    tags.forEach(tag => {
        const checkbox = document.createElement("input")
        checkbox.type = "checkbox"
        checkbox.value = tag
        checkbox.id = `tag-${tag}`

        const label = document.createElement("label")
        label.textContent = tag
        label.appendChild(checkbox)

        tagFiltersDiv.appendChild(label)
    })
}

function applyFilters() {
    const count = parseInt(document.getElementById("question-count").value) || 20
    const shuffleQuestions = document.getElementById("shuffle-questions").checked
    const tags = Array.from(document.querySelectorAll("#tag-filters input:checked")).map(checkbox => checkbox.value)
    const min = parseInt(document.getElementById("number-min").value) || -Infinity
    const max = parseInt(document.getElementById("number-max").value) || Infinity

    currentQuestions = quizQuestions.filter(q => {
        const withinNumberRange = q.number >= min && q.number <= max
        const matchesTag = tags.length === 0 || q.tags.some(tag => tags.includes(tag))
        return withinNumberRange && matchesTag
    })

    if (shuffleQuestions) shuffleArray(currentQuestions)
    currentQuestions = currentQuestions.slice(0, count)

    renderQuiz()
}

function renderQuiz() {
    const quizDiv = document.getElementById("quiz")
    quizDiv.innerHTML = ""

    currentQuestions.forEach(q => {
        const questionDiv = document.createElement("div")
        questionDiv.className = "question"

        const questionText = document.createElement("p")
        questionText.textContent = `${q.number}. ${q.question}`
        questionDiv.appendChild(questionText)

        const optionsList = document.createElement("ul")
        const options = [...q.options]
        if (document.getElementById("shuffle-options").checked) {
            shuffleArray(options)
        }
        options.forEach(option => {
            const optionItem = document.createElement("li")
            optionItem.textContent = `${option.letter}: ${option.answer}`
            optionsList.appendChild(optionItem)
        })
        questionDiv.appendChild(optionsList)

        const answerDiv = document.createElement("div")
        answerDiv.className = "answer"
        answerDiv.style.display = showAnswers ? "block" : "none"

        const answerText = document.createElement("p")
        const answer = quizAnswers.find(a => a.number === q.number)
        answerText.textContent = `Answer: ${answer.answers.map(a => a.letter).join(", ")}`
        answerDiv.appendChild(answerText)

        const explanationText = document.createElement("p")
        explanationText.textContent = `Explanation: ${answer.explanation}`
        answerDiv.appendChild(explanationText)

        questionDiv.appendChild(answerDiv)
        quizDiv.appendChild(questionDiv)
    })
}

function toggleAnswers() {
    showAnswers = !showAnswers
    document.querySelectorAll(".answer").forEach(answerDiv => {
        answerDiv.style.display = showAnswers ? "block" : "none"
    })
}

function startQuiz() {
    quizStarted = true
    document.getElementById("start-quiz").disabled = true
    document.getElementById("stop-quiz").disabled = false
}

function stopQuiz() {
    quizStarted = false
    document.getElementById("start-quiz").disabled = false
    document.getElementById("stop-quiz").disabled = true

    // Display results
    let score = 0
    currentQuestions.forEach(q => {
        const userAnswers = prompt(`Enter your answers for question ${q.number} (comma-separated):`)
        const correctAnswers = quizAnswers.find(a => a.number === q.number).answers.map(a => a.letter)
        if (
            userAnswers
                .split(",")
                .map(a => a.trim())
                .sort()
                .join(",") === correctAnswers.sort().join(",")
        ) {
            score++
        }
    })

    document.getElementById("total-score").textContent = `Your Score: ${score} / ${currentQuestions.length}`
    document.getElementById("score").style.display = "block"
}

// Event Listeners
document.getElementById("apply-filters").addEventListener("click", applyFilters)
document.getElementById("start-quiz").addEventListener("click", startQuiz)
document.getElementById("stop-quiz").addEventListener("click", stopQuiz)
document.getElementById("toggle-answers").addEventListener("click", toggleAnswers)

// Initial Setup
fetch("quiz_questions.json")
    .then(response => response.json())
    .then(data => {
        quizQuestions = data
        renderFilters(extractTags(quizQuestions))
        applyFilters()
    })

fetch("quiz_answers.json")
    .then(response => response.json())
    .then(data => {
        quizAnswers = data
    })
