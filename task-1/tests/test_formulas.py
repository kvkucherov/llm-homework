import unittest

from rag.formulas import build_formula_index, formula_sources, integral_signatures


class FormulaSearchContracts(unittest.TestCase):
    def test_general_formula_matches_renamed_variables_and_degree(self):
        blocks = [
            {"block_id": "proof", "text": r"Доказательство. $h^{(m)}(a)=\frac{m!}{2\pi i}\int_\Gamma\frac{h(t)}{(t-a)^{m+1}}dt$"},
            {"block_id": "statement", "text": r"Теорема 9.8. $g^{(k)}(b)=\frac{k!}{2\pi i}\int_\gamma\frac{g(s)}{(s-b)^{k+1}}ds$"},
        ]
        index = build_formula_index(blocks)
        for query in [r"$\oint \sin(w)/w^4\,dw$", "Интеграл sin(w)/w^4 по окружности"]:
            self.assertEqual(formula_sources(query, index), ["statement"])

    def test_nested_fraction_and_pole_order(self):
        signatures = integral_signatures(r"$\int \frac{\frac{1}{2}h(w)}{(w-a)^{7}}dw$")
        self.assertEqual([row["degree"] for row in signatures], [7])
        self.assertEqual(integral_signatures(r"$\int h(w)/w\,dw$"), [])
        self.assertEqual(integral_signatures(r"$\int h(w)/w^1\,dw$"), [])
        self.assertEqual(integral_signatures(r"$\int h(w)/(w^2+1)^4\,dw$"), [])
        self.assertEqual(integral_signatures("Производная h(w)/w^4"), [])

    def test_density_derivative_is_not_derivative_of_the_integrand(self):
        blocks = [{
            "block_id": "density",
            "text": r"Следствие 9.2. $F^{(m)}(a)=\frac{m!}{2\pi i}\int_\Gamma\frac{h(t)}{(t-a)^{m+1}}dt$",
        }]
        self.assertEqual(formula_sources(r"$\oint \sin(w)/w^4 dw$", build_formula_index(blocks)), [])


if __name__ == "__main__":
    unittest.main()
