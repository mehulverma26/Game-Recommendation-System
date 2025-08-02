document.getElementById("quiz-form").addEventListener("submit", function (e) {
  e.preventDefault();

  const form = new FormData(e.target);
  const answers = {};

  for (let [key, value] of form.entries()) {
    answers[key] = parseInt(value);
  }

  fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(answers)
  })
    .then(res => res.json())
    .then(data => {
      // redirect to result.html or dynamically display results
      localStorage.setItem("recommendations", JSON.stringify(data));
      window.location.href = "/result";
    });
});
