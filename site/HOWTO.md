# How the website works

The website is built from the Markdown files in this repository and published with GitHub Pages:
<https://kusuri26.github.io/2026B-seminar-supp/>

## One-time setup (on GitHub)

1. Push this repository to GitHub.
2. On GitHub, open **Settings → Pages**.
3. Under **Build and deployment → Source**, choose **GitHub Actions**.

After that, every push to `main` rebuilds and publishes the site (about 1 minute).
Progress and errors are shown on the **Actions** tab.

## Adding a note

1. Make a folder for the session, e.g. `20261202/` (folders named `YYYYMMDD` are shown as dates, newest first).
2. Put the Markdown file and its images in that folder. Refer to images with relative paths, e.g. `![caption](figure.png)`.
3. The first `# Heading` of the file is used as its title on the home page.
4. Commit and push.

Every `.md` file becomes a page with a table of contents; the home page lists them automatically.
The text of the root `README.md` is shown at the top of the home page.

## Writing Markdown

- Math: `$...$` inline and `$$...$$` on their own lines (rendered with MathJax), the same syntax GitHub uses.
- Section numbers are added automatically on the website: write `## Introduction`, not `## 1. Introduction` (typed numbers are replaced).
- Inside math, write `\lt` instead of `<` so GitHub does not mistake it for an HTML tag.
- Links to other notes can point to the `.md` file; they are changed to `.html` on the website.

## Previewing locally (optional)

Install [pandoc](https://pandoc.org/installing.html) and Python, then run from the repository root:

```
python site/build.py
```

and open `_site/index.html` in a browser. The `_site/` folder is not committed.

## Files

| File | Purpose |
|---|---|
| `.github/workflows/pages.yml` | GitHub Actions: build and publish on every push |
| `site/build.py` | Converts all `.md` files to HTML and makes the home page |
| `site/template.html` | Page layout (top bar, contents, MathJax) |
| `site/style.css` | Appearance: colours, fonts, layout (light and dark mode) |
| `site/links.lua` | Rewrites `.md` links to `.html` |
| `site/title.lua` | Moves the note title above the contents box |
| `site/number.lua` | Numbers the sections automatically (1., 1.1., …) |
