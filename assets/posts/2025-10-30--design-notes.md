---
title: Building a Static Blog System - Design Notes
date: 2025-10-30
tags: [web-development, javascript, github-pages, blogging, static-sites]
series: Getting Started
description: A technical deep-dive into how I built this blog system using vanilla JavaScript, GitHub Pages, and markdown files.
cover: /assets/img/2025/design-notes/architecture.png
---

# Building a Static Blog System - Design Notes

After launching this blog yesterday, several people asked about the technical implementation. Today I want to share the design decisions and architecture behind this static blog system.

## The Challenge

I wanted to create a blog that was:

- **Simple to maintain**: Adding new posts should be as easy as dropping in a markdown file
- **Fast**: No server-side rendering or database queries
- **Free to host**: Leverage GitHub Pages for hosting
- **SEO-friendly**: Proper meta tags and structure
- **Feature-rich**: Search, tags, pagination, and navigation

## Architecture Overview

The system consists of several key components:

### File Structure

```
.
├── index.html                    # Main entry point
├── assets/
│   ├── posts/                   # All markdown files
│   │   ├── 2025-10-29--hello-world.md
│   │   └── 2025-10-30--design-notes.md
│   ├── lib/                     # JavaScript and CSS
│   │   ├── blog.js
│   │   └── styles.css
│   ├── meta/                    # Configuration
│   │   └── site.json
│   └── img/                     # Images and assets
└── CNAME                        # Custom domain (optional)
```

### Frontend Components

1. **Blog System Class**: Core JavaScript that handles routing, post loading, and rendering
2. **Markdown Parser**: Converts markdown to HTML (using marked.js)
3. **Front Matter Parser**: Extracts metadata from post headers
4. **Search Engine**: Client-side search through post content
5. **Tag System**: Categorization and filtering
6. **Pagination**: Handles large numbers of posts

## Key Design Decisions

### 1. No Build Process

Instead of using a static site generator like Jekyll or Hugo, I opted for a pure client-side approach. This means:

- No need to set up build tools or dependencies
- New posts are immediately available after pushing to GitHub
- Simple deployment process
- Everything runs in the browser

### 2. Markdown File Naming Convention

Posts follow the pattern: `YYYY-MM-DD--slug.md`

This allows us to:
- Extract publish date from filename
- Generate URLs automatically
- Sort posts chronologically
- Maintain clean, readable filenames

### 3. Front Matter for Metadata

Each post starts with YAML front matter:

```yaml
---
title: Post Title
date: 2025-10-30
tags: [tag1, tag2, tag3]
series: Series Name (optional)
description: Brief description for cards and SEO
cover: /path/to/cover/image.jpg
---
```

This provides all the metadata needed for:
- Post listings
- SEO tags
- Social media cards
- Navigation
- Search indexing

### 4. Client-Side Routing

Using hash-based routing (`#/post/slug`) to:
- Enable deep linking to posts
- Support browser back/forward navigation
- Avoid 404 errors on GitHub Pages
- Keep everything in a single HTML file

## Technical Implementation

### Post Discovery

The system tries multiple methods to find posts:

1. **Index File**: If `assets/meta/index.json` exists, load all post metadata at once
2. **Auto-Discovery**: Try common filename patterns to find posts
3. **GitHub API**: Could be extended to use GitHub's API for file listing

### Content Processing

```javascript
parsePost(content, filename) {
  // Extract front matter
  const frontMatterMatch = content.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
  
  // Parse YAML metadata
  const frontMatter = this.parseFrontMatter(frontMatterMatch[1]);
  
  // Process content
  const bodyContent = frontMatterMatch[2];
  
  // Return structured post object
  return {
    slug: extractedSlug,
    title: frontMatter.title,
    date: frontMatter.date,
    content: bodyContent,
    // ... other fields
  };
}
```

### Search Implementation

Simple but effective client-side search:

```javascript
search(query) {
  const lowerQuery = query.toLowerCase();
  this.posts = this.allPosts.filter(post => 
    post.title.toLowerCase().includes(lowerQuery) ||
    post.description.toLowerCase().includes(lowerQuery) ||
    post.tags.some(tag => tag.toLowerCase().includes(lowerQuery))
  );
}
```

## Performance Considerations

### Lazy Loading

- Post content is only loaded when viewing individual posts
- List views only require metadata
- Images can be lazy-loaded as needed

### Caching

- Browser caching handles repeat visits
- LocalStorage could cache post metadata
- GitHub Pages provides ETag/Last-Modified headers

### Bundle Size

Current dependencies are minimal:
- **marked.js**: ~50KB (markdown parsing)
- **Custom CSS**: ~15KB (styling)
- **Custom JS**: ~20KB (blog logic)

Total: ~85KB plus content

## SEO and Social Media

The system automatically generates:
- Page titles with post names
- Meta descriptions from post excerpts
- Open Graph tags for social sharing
- Structured data could be added

## Future Enhancements

Some ideas for extending the system:

1. **RSS Feed**: Generate feed.xml from post metadata
2. **Sitemap**: Auto-generate sitemap.xml
3. **Comments**: Integrate with GitHub Issues or Utterances
4. **Analytics**: Add Google Analytics or privacy-focused alternatives
5. **Dark Mode**: Theme switching capability
6. **Full-Text Search**: More sophisticated search with indexing
7. **Series Navigation**: Better handling of post series

## Lessons Learned

### What Worked Well

- **Simplicity**: Easy to understand and modify
- **Performance**: Fast loading and navigation
- **Maintenance**: Adding posts is trivial
- **Flexibility**: Can easily add new features

### What Could Be Better

- **Initial Load**: Need to discover posts on first visit
- **SEO**: Limited compared to server-side rendering
- **Search**: Could be more sophisticated
- **Offline**: No service worker for offline reading

## Code and Implementation

The complete source code is available on GitHub. Key files to look at:

- `assets/lib/blog.js`: Core blog system
- `assets/lib/styles.css`: Responsive styling
- `index.html`: Main entry point
- Example posts in `assets/posts/`

## Conclusion

This static blog system proves that you don't always need complex tooling to create a functional, fast, and maintainable blog. Sometimes the simplest solution is the best one.

The entire system is under 100KB and provides a rich blogging experience with search, tags, navigation, and responsive design. Best of all, adding new content is as simple as writing a markdown file.

What do you think? Would you use a system like this for your own blog? Let me know in the comments or reach out on social media!

---

*Next up: I'll be writing about the specific CSS techniques used to create the responsive design and dark mode support.*