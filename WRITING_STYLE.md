# Writing Style Guide

This document defines the writing style for essays published on our GitHub Pages site. These essays document our cliodynamics replication project and explain our findings to a broad audience.

## Core Principles

### Write from First Principles

Begin with foundational concepts and build understanding layer by layer. Do not assume the reader has prior knowledge of cliodynamics, differential equations, or historical sociology. Every concept should be introduced from its most basic form before adding complexity.

When explaining Structural-Demographic Theory (SDT), for instance, start with the intuition that societies are systems with interacting parts, then introduce how those parts can be measured, then show how measurements change over time, and only then introduce the mathematical relationships.

### Avoid Jargon

Use plain language wherever possible. When technical terms are necessary, define them clearly on first use. The goal is accessibility without sacrificing precision.

Instead of "endogenous instability dynamics," write "instability that arises from within the system itself, rather than from outside shocks."

### Spell Out Acronyms

On first use, write the full term followed by the acronym in parentheses. After that, the acronym may be used alone.

Examples of first use:
- Structural-Demographic Theory (SDT)
- Ordinary Differential Equations (ODEs)
- Political Stress Index (PSI)
- Seshat Global History Databank (Seshat)

### Differentiate Similar Concepts

When introducing a concept, explicitly distinguish it from related concepts and from its opposite. This triangulation helps readers place new ideas in their mental map.

For example, when discussing elite overproduction, contrast it with elite scarcity, distinguish it from general population growth, and differentiate it from wealth concentration (which is related but distinct).

## Visual Integration

### Context Before and After

Every illustration, chart, and visualization must be woven into the surrounding prose. Before presenting a visual, explain what the reader should look for. After, discuss what it reveals.

The prose before a visualization should:
- State the question the visual addresses
- Explain what variables or data are shown
- Prime the reader for what patterns to notice

The prose after a visualization should:
- State the big-picture takeaway
- Acknowledge limitations of what the visual can show
- Comment on any outliers or unexpected patterns
- Connect back to the broader argument

### Comment on Outliers

When data points deviate from the expected pattern, address them explicitly. Outliers are often where the most interesting insights hide. Do not ignore them or hope readers will not notice. Instead, offer possible explanations, acknowledge uncertainty, and consider what additional evidence would help resolve the question.

### Acknowledge Limitations

Every visualization simplifies reality. Be upfront about what is not shown, what assumptions were made, and what the visualization cannot tell us. This honesty builds trust and demonstrates intellectual rigor.

## Prose Style

### Vary Rhythm

Alternate between longer, more complex sentences and shorter, direct ones. This creates a natural reading rhythm. However, avoid overusing punchy single-sentence paragraphs, which can feel choppy and gimmicky when overdone.

A paragraph might begin with a longer sentence that sets context, move through medium-length sentences that develop the idea, and conclude with a shorter sentence that lands the point. The next paragraph might reverse this pattern.

### Flowing Prose Over Lists

Avoid tables and bulleted lists in favor of connected prose. Information that might be presented as a list should instead flow as sentences within paragraphs. This style is more pleasant to read, especially when listened to with text-to-speech tools like Speechify, and it forces the writer to articulate relationships between items rather than leaving that work to the reader.

Instead of:
```
Key factors in societal instability:
- Elite overproduction
- Popular immiseration
- State fiscal crisis
```

Write:
```
Three factors drive societal instability in Turchin's framework. Elite overproduction
creates competition among those seeking positions of power and prestige. Popular
immiseration—declining living standards for ordinary people—generates a pool of
potential recruits for radical movements. And state fiscal crisis weakens the
government's ability to respond to challenges or co-opt opposition.
```

### Natural Paragraph Length

Paragraphs should vary in length. Some ideas require extended development across many sentences; others can be stated more concisely. A page with paragraphs of uniform length looks artificial. Let the content dictate the structure.

## Essay Length and Scope

### Minimum Length

Every essay must exceed one hour of reading time at 200 words per minute. This means a minimum of 12,000 words. This length is not arbitrary—it reflects our commitment to deep exploration rather than surface-level summaries.

### No Padding or Repetition

Length must come from depth, not repetition. Each paragraph should advance the argument or add new information. If you find yourself restating the same point in different words, cut back and go deeper instead.

Similarly, essays should not repeat material from other essays in the collection. Each essay occupies its own territory. Where overlap is necessary for context, refer readers to the other essay rather than reproducing its content.

### Primary Source: Our Work

The primary source material for essays is our own work in this project—the code we wrote, the data we analyzed, the models we built, the visualizations we created. Web research and external sources provide broader context and connect our work to the wider field, but they are secondary.

Essays should feel like dispatches from inside the project, not summaries of external reading.

## Dual Focus: Findings and Process

Every essay serves two purposes and should weave them together throughout.

### Documenting Findings

Essays explain what we discovered through our cliodynamics replication work:
- What the data shows
- How the models behave
- What historical patterns emerge
- How our results compare to Turchin's published work
- What new questions arise

### Documenting Process

Essays also document how we built this project:
- How Claude Code was used as the development environment
- How the worker framework distributes tasks
- How GitHub issues structure the work
- How code reviews catch problems
- How the GitHub Pages site presents our findings
- What challenges arose and how we addressed them

This meta-layer is not separate from the findings—it is woven throughout. When discussing a visualization, for instance, mention how it was generated, what code produced it, and what iterations it went through.

## Metadata Requirements

### Word Count Script

Never estimate word count or reading time. Use the `scripts/word_count.py` script to compute these values precisely. The script outputs:
- Exact word count
- Reading time at 200 words per minute
- Tags extracted from the content
- Publication date

### Required Metadata

Each essay must include in its frontmatter or header:
- Title
- Word count (computed)
- Reading time (computed)
- Publication date
- Tags for categorization

This metadata appears on the GitHub Pages index and helps readers find relevant essays.

## Voice and Tone

Write in first person plural ("we") when describing the project's work. This reflects the collaborative nature of the project between human and AI contributors.

The tone should be intellectually curious, rigorous but accessible, and honest about uncertainty. We are exploring difficult questions and do not have all the answers. That uncertainty is part of the story.

Avoid:
- Academic stuffiness
- Breathless hype
- False certainty
- Condescension toward readers

Aim for:
- Clarity
- Precision
- Intellectual honesty
- Genuine enthusiasm for the subject
