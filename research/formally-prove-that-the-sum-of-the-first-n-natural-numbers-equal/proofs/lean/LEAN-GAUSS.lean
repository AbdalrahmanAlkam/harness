/-- Sum of the first `n` natural numbers, by recursion. -/
def sumFirst : Nat → Nat
  | 0 => 0
  | n + 1 => sumFirst n + (n + 1)

/-- Distributing the product on the right twice, then commutativity. -/
theorem expand (k : Nat) : (k + 1) * ((k + 1) + 1) = k * (k + 1) + (k + 1) + (k + 1) := by
  have hA : (k + 1) * ((k + 1) + 1) = (k + 1) * (k + 1) + (k + 1) * 1 := Nat.mul_add _ _ _
  have hB : (k + 1) * 1 = k + 1 := Nat.mul_one (k + 1)
  have hC : (k + 1) * (k + 1) = (k + 1) * k + (k + 1) * 1 := Nat.mul_add (k + 1) k 1
  have hD : (k + 1) * k = k * (k + 1) := (Nat.mul_comm k (k + 1)).symm
  simp only [hA, hB, hC, hD]

/-- Gauss's sum in doubling form: twice the sum of the first `n` naturals
equals `n * (n + 1)`. -/
theorem gauss_two_mul (n : Nat) : 2 * sumFirst n = n * (n + 1) := by
  induction n with
  | zero => rfl
  | succ k ih =>
    calc 2 * (sumFirst k + (k + 1)) = 2 * sumFirst k + 2 * (k + 1) := by omega
      _ = k * (k + 1) + 2 * (k + 1) := by rw [ih]
      _ = k * (k + 1) + (k + 1) + (k + 1) := by omega
      _ = (k + 1) * ((k + 1) + 1) := (expand k).symm
      _ = (k + 1) * (k + 2) := rfl
