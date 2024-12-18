let currentQuestions = []
let showAnswers = false
let quizStarted = false
let timerInterval
let timeRemaining

function startTimer() {
    const duration = parseInt(document.getElementById("timer-duration").value)
    if (!duration) return
    timeRemaining = duration * 60 // Convert minutes to seconds
    updateTimerDisplay()
    enableTimerInput(false)

    timerInterval = setInterval(() => {
        timeRemaining--
        updateTimerDisplay()

        if (timeRemaining <= 0) {
            stopQuiz(true)
        }
    }, 1000)
}

function updateTimerDisplay() {
    const elementTimeRemaining = document.getElementById("time-remaining")
    const minutes = Math.floor(timeRemaining / 60)
    const seconds = timeRemaining % 60
    if (!timeRemaining) {
        elementTimeRemaining.textContent = ""
    } else {
        elementTimeRemaining.textContent = `${minutes}:${seconds < 10 ? "0" : ""}${seconds}`
    }
}

function enableTimerInput(enable) {
    document.getElementById("timer-duration").disabled = !enable
}

function resetTimerDuration() {
    document.getElementById("timer-duration").value = ""
}

function enableQuestions(enable) {
    const allCheckboxes = document.querySelectorAll("#quiz input[type='checkbox']")
    if (enable) {
        allCheckboxes.forEach(checkbox => (checkbox.disabled = false))
    } else {
        allCheckboxes.forEach(checkbox => (checkbox.disabled = true))
    }
}

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
        checkbox.classList.add("mr-2", "w-4", "h-4")

        const label = document.createElement("label")
        label.textContent = tag
        label.htmlFor = checkbox.id
        label.classList.add("mr-2")
        label.insertBefore(checkbox, label.firstChild)

        tagFiltersDiv.appendChild(label)
    })
}

function resetCheckboxes() {
    const allCheckboxes = document.querySelectorAll("#quiz input[type='checkbox']")
    allCheckboxes.forEach(checkbox => (checkbox.checked = false))
}

function applyFilters() {
    resetCheckboxes()
    displayScore(false)
    showAnswers = true
    toggleAnswers()

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
            optionDiv.classList.add("option-container")

            const checkbox = document.createElement("input")
            checkbox.type = "checkbox"
            checkbox.id = `${q.number}-${option.letter}`
            checkbox.value = option.letter
            checkbox.classList.add("mr-2", "w-4", "h-4")

            const label = document.createElement("label")
            label.setAttribute("for", checkbox.id)
            label.textContent = `${option.letter.toUpperCase()}) ${option.answer}`

            optionDiv.appendChild(checkbox)
            optionDiv.appendChild(label)
            optionsList.appendChild(optionDiv)
        })

        questionDiv.appendChild(optionsList)

        const answerDiv = document.createElement("div")

        answerDiv.className = "answer bg-gray-100 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg pl-2 pb-2 mt-1"
        answerDiv.style.display = showAnswers ? "block" : "none"

        // Create the feedback div and insert as first child in answerDiv
        const feedbackDiv = document.createElement("div")
        feedbackDiv.id = "isCorrect"
        answerDiv.prepend(feedbackDiv) // Insert as first child

        const answerText = document.createElement("p")

        if (q.answers.length > 1) {
            answerText.innerHTML = "<i>Answers:</i><br>" + q.answers.map(a => `${a.letter.toUpperCase()}) ${a.answer}`).join("<br>")
        } else {
            answerText.innerHTML = `<i>Answer:</i> ${q.answers[0].letter.toUpperCase()}) ${q.answers[0].answer}`
        }

        answerDiv.appendChild(answerText)

        const explanationText = document.createElement("p")
        explanationText.innerHTML = `<i>Explanation:</i> ${q.explanation}`
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

            const checkboxes = questionDiv.querySelectorAll(`input[id^="${questionNumber}-"]:checked`)
            const userAnswers = Array.from(checkboxes).map(checkbox => checkbox.value)
            const correctAnswers = quiz.find(a => a.number === parseInt(questionNumber)).answers.map(a => a.letter)

            if (userAnswers.length > 0) {
                // Check if at least one checkbox is selected
                const isCorrect = userAnswers.sort().join(",") === correctAnswers.sort().join(",")
                // Set feedback message
                feedbackDiv.textContent = isCorrect ? "Correct" : "Incorrect!"
                feedbackDiv.style.color = isCorrect ? "MediumSeaGreen" : "red"
            } else {
                feedbackDiv.textContent = "" // Clear feedback if no answer is selected
            }
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
    applyFilters()
    quizStarted = true
    const startQuizButton = document.getElementById("toggle-quiz")
    const showHideAnswersButton = document.getElementById("toggle-answers")
    startQuizButton.textContent = "Stop Quiz"
    startQuizButton.classList.add("bg-green-800", "hover:bg-dark-green-700")
    startQuizButton.classList.remove("bg-green-600", "hover:bg-green-700")
    showHideAnswersButton.disabled = true

    resetCheckboxes()
    disableFilters(true)

    showAnswers = true // Ensure answers are hidden
    toggleAnswers() // Call to hide answers if displayed
    displayScore(false)
}

function stopQuiz(forced = false) {
    quizStarted = false
    disableFilters(false)

    const startQuizButton = document.getElementById("toggle-quiz")
    const showHideAnswersButton = document.getElementById("toggle-answers")
    startQuizButton.textContent = "Start Quiz"
    startQuizButton.classList.add("bg-green-600", "hover:bg-green-700")
    startQuizButton.classList.remove("bg-green-800", "hover:bg-dark-green-700")
    showHideAnswersButton.disabled = false

    // Re-enable the "Apply Filters" button
    document.getElementById("apply-filters").disabled = false
    document.getElementById("toggle-answers").disabled = false

    displayScore(true, currentQuestions)
    clearInterval(timerInterval)
    enableTimerInput(true)
    if (forced) enableQuestions(false)
    timeRemaining = 0
    updateTimerDisplay()
    resetTimerDuration()
}

function toggleQuiz() {
    if (quizStarted) {
        stopQuiz()
    } else {
        startTimer()
        startQuiz()
    }
}

// Event Listeners
document.getElementById("apply-filters").addEventListener("click", applyFilters)
document.getElementById("toggle-quiz").addEventListener("click", toggleQuiz)
document.getElementById("toggle-answers").addEventListener("click", toggleAnswers)

renderFilters(extractTags(quiz))
applyFilters()
