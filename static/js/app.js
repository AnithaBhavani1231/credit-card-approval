const incomeInput = document.querySelector("[data-money-input]");
const loanInput = document.querySelector("[data-loan-input]");
const ratioOutput = document.querySelector("[data-ratio-output]");

function updateDebtRatio() {
  if (!incomeInput || !loanInput || !ratioOutput) {
    return;
  }

  const income = Number(incomeInput.value);
  const loan = Number(loanInput.value);
  const ratio = income > 0 ? loan / income : 0;
  ratioOutput.textContent = ratio.toFixed(2);
}

if (incomeInput && loanInput) {
  incomeInput.addEventListener("input", updateDebtRatio);
  loanInput.addEventListener("input", updateDebtRatio);
  updateDebtRatio();
}
