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

function resetCheckboxes() {
    const allCheckboxes = document.querySelectorAll("#quiz input[type='checkbox']")
    allCheckboxes.forEach(checkbox => (checkbox.checked = false))
}

function applyFilters() {
    resetCheckboxes()

    const count = parseInt(document.getElementById("question-count").value) || 20
    const shuffleQuestions = document.getElementById("shuffle-questions").checked
    const tags = Array.from(document.querySelectorAll("#tag-filters input:checked")).map(checkbox => checkbox.value)
    const min = parseInt(document.getElementById("number-min").value) || -Infinity
    const max = parseInt(document.getElementById("number-max").value) || Infinity

    currentQuestions = quiz.filter(q => {
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

        const optionsList = document.createElement("div")
        const options = [...q.options]
        if (document.getElementById("shuffle-options").checked) {
            shuffleArray(options)
        }

        options.forEach(option => {
            const optionDiv = document.createElement("div")

            const checkbox = document.createElement("input")
            checkbox.type = "checkbox"
            checkbox.id = `${q.number}-${option.letter}`
            checkbox.value = option.letter

            const label = document.createElement("label")
            label.setAttribute("for", checkbox.id)
            label.textContent = `${option.letter}: ${option.answer}`

            optionDiv.appendChild(checkbox)
            optionDiv.appendChild(label)
            optionsList.appendChild(optionDiv)
        })

        questionDiv.appendChild(optionsList)

        const answerDiv = document.createElement("div")
        answerDiv.className = "answer"
        answerDiv.style.display = showAnswers ? "block" : "none"

        // Create the feedback div and insert as first child in answerDiv
        const feedbackDiv = document.createElement("div")
        feedbackDiv.id = "isCorrect"
        answerDiv.prepend(feedbackDiv) // Insert as first child

        const answerText = document.createElement("p")
        answerText.textContent = `Answer: ${q.answers.map(a => a.letter).join(", ")}`
        answerDiv.appendChild(answerText)

        const explanationText = document.createElement("p")
        explanationText.textContent = `Explanation: ${q.explanation}`
        answerDiv.appendChild(explanationText)

        questionDiv.appendChild(answerDiv)
        quizDiv.appendChild(questionDiv)
    })
}

function toggleAnswers() {
    showAnswers = !showAnswers
    document.querySelectorAll(".answer").forEach(answerDiv => {
        answerDiv.style.display = showAnswers ? "block" : "none"

        // Get the feedback div specific to this question
        const feedbackDiv = answerDiv.parentElement.querySelector("#isCorrect")

        if (showAnswers) {
            // Check user-selected answers for each question
            const questionDiv = answerDiv.parentElement
            const questionNumber = questionDiv.querySelector("p").textContent.split(".")[0] // Get the question number

            const userAnswers = []
            const checkboxes = questionDiv.querySelectorAll(`input[id^="${questionNumber}-"]:checked`)
            checkboxes.forEach(checkbox => userAnswers.push(checkbox.value))

            const correctAnswers = quiz.find(a => a.number === parseInt(questionNumber)).answers.map(a => a.letter)
            const isCorrect = userAnswers.sort().join(",") === correctAnswers.sort().join(",")

            // Set feedback message
            feedbackDiv.textContent = isCorrect ? "Correct" : "Incorrect!"
            feedbackDiv.style.color = isCorrect ? "MediumSeaGreen" : "red"
        } else {
            // Clear feedback when answers are hidden
            feedbackDiv.textContent = ""
        }
    })
}

function displayScore(display, currentQuestions) {
    if (!display) {
        document.getElementById("score").style.display = "none"
    } else {
        let score = 0

        currentQuestions.forEach(q => {
            const userAnswers = []
            const checkboxes = document.querySelectorAll(`input[id^="${q.number}-"]:checked`)
            checkboxes.forEach(checkbox => userAnswers.push(checkbox.value))

            const correctAnswers = quiz.find(a => a.number === q.number).answers.map(a => a.letter)
            if (userAnswers.sort().join(",") === correctAnswers.sort().join(",")) {
                score++
            }
        })

        document.getElementById("total-score").textContent = `Your Score: ${score} / ${currentQuestions.length}`
        document.getElementById("score").style.display = "block"
    }
}

function disableFilters(disabled) {
    const filterElements = document.querySelectorAll("#filters input, #filters button")
    filterElements.forEach(element => {
        element.disabled = disabled
    })
}

function startQuiz() {
    quizStarted = true
    document.getElementById("toggle-quiz").textContent = "Stop Quiz" // Change to Stop Quiz

    resetCheckboxes()
    disableFilters(true)

    // Hide answers and score display
    showAnswers = true // Ensure answers are hidden
    toggleAnswers() // Call to hide answers if displayed
    // document.getElementById("score").style.display = "none"
    displayScore(false)
}

function stopQuiz() {
    quizStarted = false
    disableFilters(false)

    document.getElementById("toggle-quiz").textContent = "Start Quiz" // Change back to Start Quiz

    // Re-enable the "Apply Filters" button
    document.getElementById("apply-filters").disabled = false
    document.getElementById("toggle-answers").disabled = false

    displayScore(true, currentQuestions)
}

function toggleQuiz() {
    if (quizStarted) {
        stopQuiz()
        document.getElementById("toggle-quiz").textContent = "Start Quiz" // Change to Start Quiz
    } else {
        startQuiz()
        document.getElementById("toggle-quiz").textContent = "Stop Quiz" // Change to Stop Quiz
    }
}

// Event Listeners
document.getElementById("apply-filters").addEventListener("click", applyFilters)
document.getElementById("toggle-quiz").addEventListener("click", toggleQuiz)
document.getElementById("toggle-answers").addEventListener("click", toggleAnswers)

renderFilters(extractTags(quiz))
applyFilters()
