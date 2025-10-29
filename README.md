# Static Blog System for GitHub Pages

A lightweight, fully client-side blog system built for GitHub Pages. No build process required - just add markdown files and go!

![Blog Preview](https://via.placeholder.com/800x400/2563eb/ffffff?text=Your+Blog+Preview)

## ✨ Features

- 📝 **Markdown-based posting** - Write in markdown with YAML front matter
- 🏷️ **Tag system** - Organize posts by topics
- 📅 **Date-based archives** - Browse posts by year and month  
- 🔍 **Client-side search** - Find posts by title, content, or tags
- 📱 **Responsive design** - Works great on mobile and desktop
- 🚀 **No build process** - Direct deployment to GitHub Pages
- ⚡ **Fast loading** - Optimized for performance
- 🎨 **Customizable** - Easy to modify colors, fonts, and layout

## 🚀 Quick Start

### 1. Use This Template

Click "Use this template" to create your own repository, or fork this repository.

### 2. Enable GitHub Pages

1. Go to your repository settings
2. Scroll to "Pages" section
3. Set source to "Deploy from a branch"
4. Choose "main" branch and "/ (root)" folder
5. Save and wait for deployment

### 3. Customize Your Blog

Edit `assets/meta/site.json` with your information:

```json
{
  "title": "Your Blog Name",
  "subtitle": "Your subtitle here",
  "description": "Your blog description",
  "author": {
    "name": "Your Name",
    "email": "your.email@example.com",
    "bio": "Your bio"
  },
  "url": "https://yourusername.github.io",
  "social": {
    "github": "yourusername",
    "twitter": "yourusername",
    "linkedin": "yourusername"
  }
}
```

### 4. Add Your First Post

Create a new file in `assets/posts/` following the naming pattern: `YYYY-MM-DD--slug.md`

Example: `assets/posts/2025-10-29--my-first-post.md`

```markdown
---
title: My First Blog Post
date: 2025-10-29
tags: [welcome, introduction]
description: This is my first post on my new blog!
---

# My First Blog Post

Welcome to my blog! This is where I'll share my thoughts and experiences.

## What's Next

I plan to write about:
- Technology and programming
- Personal projects
- Life experiences
- And much more!
```

### 5. Push and Deploy

Commit your changes and push to the main branch. GitHub Pages will automatically deploy your blog!

## 📁 Project Structure

```
.
├── index.html                    # Main blog page
├── tags.html                     # Tags overview page  
├── archives.html                 # Date-based archives
├── assets/
│   ├── posts/                   # Your blog posts (markdown files)
│   │   ├── 2025-10-29--hello-world.md
│   │   └── 2025-10-30--second-post.md
│   ├── lib/                     # JavaScript and CSS
│   │   ├── blog.js             # Core blog functionality
│   │   └── styles.css          # Styling
│   ├── meta/                    # Configuration files
│   │   └── site.json           # Site settings
│   ├── img/                     # Images for posts
│   └── files/                   # Downloadable files
├── CNAME                        # Custom domain (optional)
└── README.md                    # This file
```

## ✍️ Writing Posts

### File Naming Convention

Posts must follow this naming pattern: `YYYY-MM-DD--slug.md`

- `YYYY-MM-DD`: Publication date
- `slug`: URL-friendly identifier (lowercase, hyphens only)

Examples:
- `2025-01-15--hello-world.md`
- `2025-02-03--javascript-tips.md`
- `2025-03-10--project-update.md`

### Front Matter

Each post should start with YAML front matter:

```yaml
---
title: Your Post Title                    # Required
date: YYYY-MM-DD                         # Required (or extracted from filename)
tags: [tag1, tag2, tag3]                # Optional
series: Series Name                       # Optional
description: Brief description            # Optional (used for excerpts and SEO)
cover: /assets/img/2025/post/image.jpg   # Optional
---
```

### Content Guidelines

- Use standard markdown syntax
- Images should be placed in `assets/img/YYYY/post-slug/`
- Reference images with relative paths: `![Alt text](../img/2025/post-slug/image.jpg)`
- Use meaningful alt text for accessibility
- Keep line length reasonable for readability

### Example Post Structure

```markdown
---
title: How to Build Amazing Web Apps
date: 2025-10-29
tags: [web-development, javascript, tutorial]
description: A comprehensive guide to building modern web applications
cover: /assets/img/2025/web-apps/hero.jpg
---

# How to Build Amazing Web Apps

Introduction paragraph that hooks the reader...

## Section 1: Getting Started

Content for the first section...

### Subsection

More detailed content...

## Section 2: Advanced Techniques

```javascript
// Code example
function buildApp() {
  return "awesome";
}
```

## Conclusion

Wrap up your thoughts...
```

## 🎨 Customization

### Changing Colors and Fonts

Edit the CSS variables in `assets/lib/styles.css`:

```css
:root {
  --primary-color: #2563eb;        /* Main brand color */
  --accent-color: #059669;         /* Secondary color */
  --font-family: 'Inter', sans-serif;
}
```

### Adding Custom Pages

Create new HTML files in the root directory and add them to the navigation in each page's header.

### Modifying Layout

The main layout logic is in `assets/lib/blog.js`. Key functions:
- `renderPostCard()`: How posts appear in lists
- `showPost()`: Individual post display
- `renderMarkdown()`: Markdown to HTML conversion

## 🔧 Advanced Configuration

### Custom Domain

1. Add a `CNAME` file to the repository root with your domain
2. Configure DNS settings with your domain provider
3. Update the `url` field in `assets/meta/site.json`

### Analytics

Add analytics code to the `<head>` section of HTML files:

```html
<!-- Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'GA_MEASUREMENT_ID');
</script>
```

### Comments

Integrate with services like:
- [Utterances](https://utteranc.es/) (GitHub Issues-based)
- [Disqus](https://disqus.com/)
- [Giscus](https://giscus.app/) (GitHub Discussions-based)

### RSS Feed

To add RSS functionality, create a GitHub Action that generates `feed.xml` from your posts.

## 📊 SEO and Performance

### Built-in SEO Features

- Semantic HTML structure
- Meta descriptions from post front matter
- Open Graph tags for social sharing
- Proper heading hierarchy
- Clean URLs with meaningful slugs

### Performance Optimizations

- Minimal JavaScript (~20KB core + libraries)
- CSS optimized for fast loading
- Lazy loading for post content
- Browser caching via GitHub Pages
- Responsive images support

### Best Practices

1. **Write descriptive titles** - Good for SEO and user experience
2. **Use meaningful meta descriptions** - Shows in search results
3. **Optimize images** - Compress and use appropriate formats
4. **Internal linking** - Link between related posts
5. **Regular posting** - Fresh content improves search ranking

## 🐛 Troubleshooting

### Posts Not Showing Up

1. Check file naming convention: `YYYY-MM-DD--slug.md`
2. Verify front matter syntax (valid YAML)
3. Ensure file is in `assets/posts/` directory
4. Check browser console for JavaScript errors

### Styling Issues

1. Clear browser cache
2. Check CSS syntax in custom modifications
3. Verify file paths are correct
4. Test on different devices/browsers

### GitHub Pages Not Updating

1. Check repository settings for Pages configuration
2. Verify latest changes are pushed to main branch
3. Wait 5-10 minutes for deployment
4. Check GitHub Actions tab for build status

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [Marked.js](https://marked.js.org/) for markdown parsing
- [Highlight.js](https://highlightjs.org/) for syntax highlighting
- [Font Awesome](https://fontawesome.com/) for icons
- [Inter Font](https://rsms.me/inter/) for typography

## 📞 Support

If you encounter issues or have questions:

1. Check this README for common solutions
2. Search existing [GitHub Issues](https://github.com/yourusername/yourrepo/issues)
3. Create a new issue with detailed information
4. Join discussions in [GitHub Discussions](https://github.com/yourusername/yourrepo/discussions)

---

**Happy blogging!** 🎉

Made with ❤️ for the developer community.