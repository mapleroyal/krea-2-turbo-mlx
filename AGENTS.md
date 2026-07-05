# General

- NO SHORTCUTS. If the ideal route requires a certain amount of work, do that work. Do things the right way the first time. If you find yourself taking a shortcut or come across existing work that feels like a shortcut, stop and ask if there's a strong justification, or if it compromises the quality of the project. If the answer is yes, clearly report this to the human.

# Documentation

- Put information in its most appropriate single location. If the information is genuinely required elsewhere, refer to its location instead of repeating it. We want to dedupe and limit redundancy.

# Troubleshooting

- **Root-Cause Fixes Only**: Diagnose and correct the causes of problems, not their symptoms; avoid workarounds. If there is any uncertainty, eagerly search the web for the most current information.
- **Log Before You Leap**: When the correct solution isn't obvious, speculative attempts are allowed, but you MUST use logging to trace the issue.
- **Cleanup**: Whenever an attempt doesn't work, remove it before trying the next one.

# Tests

- **High-signal tests only**: Tests should protect meaningful behavior, contracts, edge cases, or known regressions; avoid tests that merely freeze implementation details, mocked plumbing, or trivial rendering. **NO TESTS JUST TO INCREASE COVERAGE.**

# Reinventing the Wheel

- If a solution/implementation isn't obvious, search the web to see patterns from dominant players in the space rather than walking the path of trial and error.

# Source Reuse

- Reuse proven, existing MLX implementations when they exactly satisfy a project requirement. If reuse fails, document the concrete gap before implementing custom code.

# Scope Control

- Don't work ahead of the current task. Let the implementers of future tasks own the design and implementation of those tasks.
- You may repair or improve work from before your task if it's tied to your task.
- Never build backwards/legacy compatibility. The project is in active pre-release development.
