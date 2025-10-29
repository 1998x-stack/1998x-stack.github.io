// Blog system main JavaScript
class BlogSystem {
  constructor() {
    this.siteConfig = null;
    this.posts = [];
    this.allPosts = [];
    this.currentPage = 1;
    this.postsPerPage = 10;
    this.currentTag = null;
    this.currentYear = null;
    this.currentMonth = null;
    this.searchQuery = '';
    
    this.init();
  }

  async init() {
    try {
      await this.loadSiteConfig();
      await this.loadPosts();
      this.initializeRouting();
      this.bindEvents();
    } catch (error) {
      console.error('Failed to initialize blog system:', error);
      this.showError('Failed to load blog. Please try again later.');
    }
  }

  async loadSiteConfig() {
    try {
      const response = await fetch('./assets/meta/site.json');
      this.siteConfig = await response.json();
      this.postsPerPage = this.siteConfig.pagination?.postsPerPage || 10;
      this.updateSiteTitle();
    } catch (error) {
      console.error('Failed to load site config:', error);
      this.siteConfig = this.getDefaultSiteConfig();
    }
  }

  getDefaultSiteConfig() {
    return {
      title: 'My Blog',
      subtitle: 'A personal blog',
      description: 'Personal thoughts and experiences',
      author: { name: 'Anonymous' }
    };
  }

  async loadPosts() {
    try {
      // Try to load from index.json first (if exists)
      try {
        const response = await fetch('./assets/meta/index.json');
        if (response.ok) {
          this.allPosts = await response.json();
          this.posts = [...this.allPosts];
          return;
        }
      } catch (e) {
        // Fall back to discovering posts
      }

      // Discover posts by trying common filenames
      await this.discoverPosts();
    } catch (error) {
      console.error('Failed to load posts:', error);
      this.showError('Failed to load posts.');
    }
  }

  async discoverPosts() {
    // This is a simplified discovery method
    // In production, you might want to use GitHub API or generate index.json
    const posts = [];
    const currentYear = new Date().getFullYear();
    
    // Try to discover posts from the last 2 years
    for (let year = currentYear; year >= currentYear - 1; year--) {
      for (let month = 12; month >= 1; month--) {
        for (let day = 31; day >= 1; day--) {
          const dateStr = `${year}-${month.toString().padStart(2, '0')}-${day.toString().padStart(2, '0')}`;
          
          // Try common slug patterns
          const commonSlugs = ['hello-world', 'first-post', 'welcome', 'introduction'];
          for (const slug of commonSlugs) {
            try {
              const filename = `${dateStr}--${slug}.md`;
              const response = await fetch(`./assets/posts/${filename}`);
              if (response.ok) {
                const content = await response.text();
                const post = this.parsePost(content, filename);
                if (post) {
                  posts.push(post);
                }
              }
            } catch (e) {
              // Continue trying
            }
          }
        }
      }
    }

    this.allPosts = posts.sort((a, b) => new Date(b.date) - new Date(a.date));
    this.posts = [...this.allPosts];
  }

  parsePost(content, filename) {
    const frontMatterMatch = content.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
    
    if (!frontMatterMatch) {
      return null;
    }

    const frontMatter = this.parseFrontMatter(frontMatterMatch[1]);
    const bodyContent = frontMatterMatch[2];
    
    // Extract date and slug from filename
    const filenameMatch = filename.match(/(\d{4}-\d{2}-\d{2})--(.+)\.md$/);
    const fileDate = filenameMatch ? filenameMatch[1] : null;
    const fileSlug = filenameMatch ? filenameMatch[2] : filename.replace('.md', '');

    const post = {
      slug: fileSlug,
      title: frontMatter.title || this.extractTitleFromContent(bodyContent) || 'Untitled',
      date: frontMatter.date || fileDate || new Date().toISOString().split('T')[0],
      tags: frontMatter.tags || [],
      series: frontMatter.series || null,
      description: frontMatter.description || this.extractExcerpt(bodyContent),
      cover: frontMatter.cover || null,
      content: bodyContent,
      filename: filename,
      readingTime: this.calculateReadingTime(bodyContent)
    };

    return post;
  }

  parseFrontMatter(yamlContent) {
    // Use js-yaml if available for better parsing
    if (typeof jsyaml !== 'undefined') {
      try {
        return jsyaml.load(yamlContent) || {};
      } catch (error) {
        console.error('Error parsing YAML front matter:', error);
        // Fall back to simple parsing
      }
    }
    
    // Simple fallback YAML parsing
    const result = {};
    const lines = yamlContent.split('\n');
    
    for (const line of lines) {
      const match = line.match(/^([^:]+):\s*(.*)$/);
      if (match) {
        const key = match[1].trim();
        let value = match[2].trim();
        
        // Handle arrays (tags)
        if (value.startsWith('[') && value.endsWith(']')) {
          value = value.slice(1, -1).split(',').map(s => s.trim().replace(/['"]/g, ''));
        } else {
          // Remove quotes
          value = value.replace(/^['"]|['"]$/g, '');
        }
        
        result[key] = value;
      }
    }
    
    return result;
  }

  extractTitleFromContent(content) {
    const match = content.match(/^#\s+(.+)$/m);
    return match ? match[1] : null;
  }

  extractExcerpt(content, maxLength = 160) {
    // Remove markdown syntax and get plain text
    const plainText = content
      .replace(/^#.*$/gm, '') // Remove headers
      .replace(/\*\*(.*?)\*\*/g, '$1') // Remove bold
      .replace(/\*(.*?)\*/g, '$1') // Remove italic
      .replace(/`(.*?)`/g, '$1') // Remove inline code
      .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1') // Remove links
      .replace(/\n/g, ' ')
      .trim();
    
    return plainText.length > maxLength 
      ? plainText.substring(0, maxLength) + '...'
      : plainText;
  }

  calculateReadingTime(content) {
    const wordsPerMinute = 200;
    const words = content.split(/\s+/).length;
    const minutes = Math.ceil(words / wordsPerMinute);
    return `${minutes} min read`;
  }

  initializeRouting() {
    const hash = window.location.hash;
    
    if (hash.startsWith('#/post/')) {
      const slug = hash.replace('#/post/', '');
      this.showPost(slug);
    } else if (hash.startsWith('#/tag/')) {
      const tag = hash.replace('#/tag/', '');
      this.filterByTag(tag);
    } else if (hash.startsWith('#/archives')) {
      this.showArchives(hash);
    } else if (hash.startsWith('#/search')) {
      const params = new URLSearchParams(hash.split('?')[1]);
      const query = params.get('q') || '';
      this.search(query);
    } else {
      this.showPostList();
    }
  }

  bindEvents() {
    // Search input
    const searchInput = document.getElementById('search-input');
    if (searchInput) {
      searchInput.addEventListener('input', (e) => {
        this.search(e.target.value);
      });
    }

    // Tag filter
    const tagFilter = document.getElementById('tag-filter');
    if (tagFilter) {
      tagFilter.addEventListener('change', (e) => {
        if (e.target.value) {
          this.filterByTag(e.target.value);
        } else {
          this.clearFilters();
        }
      });
    }

    // Hash change for routing
    window.addEventListener('hashchange', () => {
      this.initializeRouting();
    });

    // Back to top button
    window.addEventListener('scroll', () => {
      const backToTop = document.getElementById('back-to-top');
      if (backToTop) {
        backToTop.style.display = window.scrollY > 500 ? 'block' : 'none';
      }
    });
  }

  updateSiteTitle(pageTitle = null) {
    const baseTitle = this.siteConfig.title;
    document.title = pageTitle ? `${pageTitle} - ${baseTitle}` : baseTitle;
  }

  showPostList(page = 1) {
    this.currentPage = page;
    
    const startIndex = (page - 1) * this.postsPerPage;
    const endIndex = startIndex + this.postsPerPage;
    const postsToShow = this.posts.slice(startIndex, endIndex);
    
    const content = document.getElementById('content');
    if (!content) return;

    const html = `
      <div class="posts-section">
        <div class="search-controls">
          <input type="text" id="search-input" class="search-input" placeholder="Search posts..." value="${this.searchQuery}">
          <select id="tag-filter" class="filter-select">
            <option value="">All Tags</option>
            ${this.getAllTags().map(tag => 
              `<option value="${tag}" ${this.currentTag === tag ? 'selected' : ''}>${tag}</option>`
            ).join('')}
          </select>
        </div>
        
        ${this.currentTag || this.searchQuery ? `
          <div class="active-filters">
            ${this.currentTag ? `<span class="filter-tag">Tag: ${this.currentTag} <button onclick="blog.clearFilters()">&times;</button></span>` : ''}
            ${this.searchQuery ? `<span class="filter-tag">Search: "${this.searchQuery}" <button onclick="blog.clearFilters()">&times;</button></span>` : ''}
          </div>
        ` : ''}

        <div class="posts-grid">
          ${postsToShow.map(post => this.renderPostCard(post)).join('')}
        </div>

        ${this.posts.length === 0 ? '<div class="empty-state">No posts found matching your criteria.</div>' : ''}

        ${this.renderPagination()}
      </div>
    `;

    content.innerHTML = html;
    this.bindEvents();
    this.updateSiteTitle();
  }

  renderPostCard(post) {
    return `
      <article class="post-card">
        <div class="post-card-header">
          <h2 class="post-title">
            <a href="#/post/${post.slug}">${post.title}</a>
          </h2>
          <div class="post-meta">
            <span class="post-date">📅 ${this.formatDate(post.date)}</span>
            <span class="post-reading-time">⏱️ ${post.readingTime}</span>
          </div>
        </div>
        
        ${post.description ? `<p class="post-excerpt">${post.description}</p>` : ''}
        
        ${post.tags.length > 0 ? `
          <div class="post-tags">
            ${post.tags.map(tag => `<a href="#/tag/${tag}" class="tag">${tag}</a>`).join('')}
          </div>
        ` : ''}
      </article>
    `;
  }

  renderPagination() {
    const totalPages = Math.ceil(this.posts.length / this.postsPerPage);
    
    if (totalPages <= 1) return '';

    let pagination = '<div class="pagination">';
    
    // Previous button
    if (this.currentPage > 1) {
      pagination += `<a href="#" onclick="blog.showPostList(${this.currentPage - 1})">&laquo; Previous</a>`;
    }
    
    // Page numbers
    for (let i = 1; i <= totalPages; i++) {
      if (i === this.currentPage) {
        pagination += `<span class="current">${i}</span>`;
      } else {
        pagination += `<a href="#" onclick="blog.showPostList(${i})">${i}</a>`;
      }
    }
    
    // Next button
    if (this.currentPage < totalPages) {
      pagination += `<a href="#" onclick="blog.showPostList(${this.currentPage + 1})">Next &raquo;</a>`;
    }
    
    pagination += '</div>';
    return pagination;
  }

  async showPost(slug) {
    const post = this.allPosts.find(p => p.slug === slug);
    
    if (!post) {
      this.showError('Post not found.');
      return;
    }

    let content = post.content;
    
    // If we don't have the full content, load it
    if (!content) {
      try {
        const response = await fetch(`./assets/posts/${post.filename}`);
        const fullContent = await response.text();
        const parsedPost = this.parsePost(fullContent, post.filename);
        content = parsedPost.content;
      } catch (error) {
        this.showError('Failed to load post content.');
        return;
      }
    }

    const contentDiv = document.getElementById('content');
    if (!contentDiv) return;

    const renderedContent = this.renderMarkdown(content);
    const navigation = this.getPostNavigation(slug);

    const html = `
      <article class="article">
        <header class="article-header">
          <h1 class="article-title">${post.title}</h1>
          <div class="article-meta">
            <span class="post-date">📅 ${this.formatDate(post.date)}</span>
            <span class="post-reading-time">⏱️ ${post.readingTime}</span>
          </div>
          ${post.tags.length > 0 ? `
            <div class="article-tags">
              ${post.tags.map(tag => `<a href="#/tag/${tag}" class="tag">${tag}</a>`).join('')}
            </div>
          ` : ''}
        </header>
        
        <div class="article-content">
          ${renderedContent}
        </div>
      </article>

      ${navigation}
    `;

    contentDiv.innerHTML = html;
    this.updateSiteTitle(post.title);
    
    // Scroll to top
    window.scrollTo(0, 0);
  }

  getPostNavigation(currentSlug) {
    const currentIndex = this.allPosts.findIndex(p => p.slug === currentSlug);
    if (currentIndex === -1) return '';

    const prevPost = this.allPosts[currentIndex + 1];
    const nextPost = this.allPosts[currentIndex - 1];

    if (!prevPost && !nextPost) return '';

    return `
      <nav class="post-navigation">
        ${prevPost ? `
          <a href="#/post/${prevPost.slug}" class="nav-post prev">
            <div class="nav-post-label">Previous Post</div>
            <div class="nav-post-title">${prevPost.title}</div>
          </a>
        ` : '<div></div>'}
        
        ${nextPost ? `
          <a href="#/post/${nextPost.slug}" class="nav-post next">
            <div class="nav-post-label">Next Post</div>
            <div class="nav-post-title">${nextPost.title}</div>
          </a>
        ` : '<div></div>'}
      </nav>
    `;
  }

  renderMarkdown(content) {
    // Use marked.js if available, otherwise fall back to simple parsing
    if (typeof marked !== 'undefined') {
      // Configure marked for better security and features
      marked.setOptions({
        gfm: true,
        breaks: false,
        pedantic: false,
        sanitize: false,
        smartLists: true,
        smartypants: false,
        highlight: function(code, lang) {
          // Use highlight.js if available
          if (typeof hljs !== 'undefined' && lang && hljs.getLanguage(lang)) {
            try {
              return hljs.highlight(code, { language: lang }).value;
            } catch (err) {
              return code;
            }
          }
          return code;
        }
      });
      
      return marked.parse(content);
    }
    
    // Fallback simple markdown rendering
    return content
      .replace(/^### (.*$)/gm, '<h3>$1</h3>')
      .replace(/^## (.*$)/gm, '<h2>$1</h2>')
      .replace(/^# (.*$)/gm, '<h1>$1</h1>')
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/`(.*?)`/g, '<code>$1</code>')
      .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>')
      .replace(/^> (.*$)/gm, '<blockquote>$1</blockquote>')
      .replace(/\n\n/g, '</p><p>')
      .replace(/^/, '<p>')
      .replace(/$/, '</p>')
      .replace(/<p><\/p>/g, '')
      .replace(/<p>(<h[1-6]>)/g, '$1')
      .replace(/(<\/h[1-6]>)<\/p>/g, '$1')
      .replace(/<p>(<blockquote>)/g, '$1')
      .replace(/(<\/blockquote>)<\/p>/g, '$1');
  }

  filterByTag(tag) {
    this.currentTag = tag;
    this.posts = this.allPosts.filter(post => post.tags.includes(tag));
    this.currentPage = 1;
    this.showPostList();
    window.location.hash = `#/tag/${tag}`;
  }

  search(query) {
    this.searchQuery = query.trim();
    
    if (!this.searchQuery) {
      this.posts = [...this.allPosts];
    } else {
      const lowerQuery = this.searchQuery.toLowerCase();
      this.posts = this.allPosts.filter(post => 
        post.title.toLowerCase().includes(lowerQuery) ||
        post.description.toLowerCase().includes(lowerQuery) ||
        post.tags.some(tag => tag.toLowerCase().includes(lowerQuery))
      );
    }
    
    this.currentPage = 1;
    this.showPostList();
    
    if (this.searchQuery) {
      window.location.hash = `#/search?q=${encodeURIComponent(this.searchQuery)}`;
    }
  }

  clearFilters() {
    this.currentTag = null;
    this.searchQuery = '';
    this.posts = [...this.allPosts];
    this.currentPage = 1;
    this.showPostList();
    window.location.hash = '#/';
  }

  getAllTags() {
    const tags = new Set();
    this.allPosts.forEach(post => {
      post.tags.forEach(tag => tags.add(tag));
    });
    return Array.from(tags).sort();
  }

  formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { 
      year: 'numeric', 
      month: 'long', 
      day: 'numeric' 
    });
  }

  showError(message) {
    const content = document.getElementById('content');
    if (content) {
      content.innerHTML = `<div class="error">${message}</div>`;
    }
  }

  showLoading() {
    const content = document.getElementById('content');
    if (content) {
      content.innerHTML = '<div class="loading">Loading...</div>';
    }
  }
}

// Initialize the blog when DOM is loaded
let blog;
document.addEventListener('DOMContentLoaded', () => {
  blog = new BlogSystem();
});

// Make blog instance globally available for onclick handlers
window.blog = blog;