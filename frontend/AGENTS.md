# Repository Guidelines

## Supreme Top Priority — The "Codebase Health" Rule

Treat every contribution as a long-term architectural decision: follow established patterns, conventions, and ownership boundaries. Where sensible from a high-level, long-term perspective, extend existing constructs before introducing new ones or one-off paths. Each change should leave the codebase more coherent than before by centralizing responsibility, avoiding parallel implementations, and delivering net simplification rather than net complexity.

**IMPORTANT: At the beginning of your very first response to the human in a new conversation, state your awareness of the above rule and repeat it ver batim.**

(We established this after a costly rewrite caused by reactive, directionless/ad-hoc coding. Regardless of size or scope, every change must pass this big-picture test: "If the app were designed from scratch to include this, is this how the requirement would be implemented?")

## UI Components

**Check before building**: Use `@/components/ui/*` and Shadcn/Radix primitives. Compose from existing parts when possible. Build from scratch only when none of the above satisfy the requirement.

## UI Implementation Policy

- For any screen, modal, or flow, begin with modern default implementations and canonical library patterns. Only introduce customization after a working default is in place and there is a clear product requirement to diverge, or else when approved by (or requested by) the human.
- There is a typography design system at `docs/learnings/typography-design-system.md`. Follow it.

## Styling Policy

- Use Tailwind v4. If you aren’t sure, reference the v4 docs.
- **Design Direction**: Minimal/simple, elegant, avoid copy except where it's essential.

## Debugging & Problem-Solving

- **Root-Cause Fixes Only**: Diagnose and correct the causes of problems, not their symptoms; avoid workarounds or patching established, likely-stable packages. Prefer fixes that are idiomatic to the stack components involved. If there is any uncertainty, eagerly search the docs on the web for the most current best practices.
- **Log Before You Leap**: When the correct solution isn't obvious, add console logging to trace actual runtime behavior rather than guessing and applying speculative fixes.

## General

- **DRY**: For all changes, big or small, local or systemic, ask yourself, "Does something else in the codebase do the same thing, or even something similar?" If yes, consider whether DRYing it up would be appropriate.
- For any feature without built-in support from the tech stack, prefer (and install, if needed) the overwhelmingly popular/dominant library/package over hacking together a custom implementation.
- Never build backwards/legacy compatibility. The app is in active pre-release development.
