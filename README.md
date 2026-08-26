# Source Code and Assets of kisnikser.github.io

This repository contains the source code and assets of Nikita Kiselev's website at https://kisnikser.github.io/.

## Source code

The website is built with [Hugo](https://gohugo.io/) and the [PaperMod](https://github.com/adityatelange/hugo-PaperMod) theme (git submodule in `themes/PaperMod`).
The layout and styles were originally derived from the minimalist template at [pmichaillat/hugo-website](https://github.com/pmichaillat/hugo-website).

## Local development

```bash
git clone --recurse-submodules https://github.com/kisnikser/kisnikser.github.io.git
cd kisnikser.github.io
# if the repository was cloned without submodules:
git submodule update --init

hugo server            # live preview at http://localhost:1313/
hugo --gc --minify     # production build into ./public
```

Requirements: Hugo **extended** edition (needed for WebP image processing), version 0.128 or newer.
The GitHub Actions workflow (`.github/workflows/deploy.yml`) pins the exact version used for deployment.
Pushes to `main` are deployed to GitHub Pages; pull requests are only built.

## Layout

| Path | Purpose |
| --- | --- |
| `config.yml` | Site configuration (menu, profile, social links, theme flags) |
| `content/publications/`, `content/talks/`, `content/projects/` | One page bundle (`<slug>/index.md` + images/PDFs) per item |
| `layouts/` | Overrides of PaperMod templates; only files that actually differ from the theme are kept here |
| `layouts/partials/extend_head.html` | Hook for site-specific `<head>` additions (KaTeX) |
| `layouts/_default/_markup/render-image.html`, `layouts/partials/cover.html` | Automatic WebP + `srcset` generation for images |
| `assets/css/` | Style overrides (concatenated with the theme's CSS) |
| `assets/photo.jpg` | Profile photo; resized at build time |
| `static/` | Files published as-is (PDFs, favicons) |
| `archetypes/` | Front matter templates for `hugo new` |

## Images

Put images next to the page's `index.md` and reference them as usual:

```markdown
![](figure.png)
```

At build time Hugo generates WebP variants (up to 1440 px wide) and a `srcset` for them, so large source files can be committed as they are.
The original file is still published and can be linked directly.
Formulas are rendered with KaTeX on pages that set `math: true` in their front matter.

## Assets

The website's assets include [publications](https://kisnikser.github.io/publications/), [talks](https://kisnikser.github.io/talks/), and [projects](https://kisnikser.github.io/projects/).
They are stored in the `content` folder (Markdown, images, PDFs) and `static` folder (resume, CV) and can be batch downloaded from there.

## Tips

> [!TIP]
> You may face a tricky problem with LaTeX, i.e. some formulas cannot be correctly viewed on the website.
> Check [this](https://kiwamizamurai.github.io/posts/2022-03-06/#problem).

## License

Except where otherwise noted, the website's content was created by Nikita Kiselev and is licensed under the [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/).
