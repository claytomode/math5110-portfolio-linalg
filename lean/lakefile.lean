import Lake
open Lake DSL

package «portfolio-linalg» where
  leanOptions := #[
    ⟨`autoImplicit, false⟩,
    ⟨`relaxedAutoImplicit, false⟩
  ]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.19.0"

@[default_target]
lean_lib PortfolioLinAlg where
  roots := #[`PortfolioLinAlg]
