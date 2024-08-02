// 当前页数
let currentPage = 1;

// 最大页数（根据实际存在的页面数量设置）
const maxPage = 2;

// 加载指定页码的内容
function loadPage(page) {
    if (page < 1 || page > maxPage) {
        console.error('Invalid page number:', page);
        return;
    }

    fetch(`HOME/${page}.html`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.text();
        })
        .then(data => {
            document.getElementById('blog-content').innerHTML = data;
        })
        .catch(error => {
            console.error('Error fetching the page:', error);
        });
}

// 初始化加载第一页
document.addEventListener('DOMContentLoaded', () => {
    loadPage(currentPage);
});

// 上一页按钮点击事件
document.getElementById('prev-page').addEventListener('click', () => {
    if (currentPage > 1) {
        currentPage--;
        loadPage(currentPage);
    }
});

// 下一页按钮点击事件
document.getElementById('next-page').addEventListener('click', () => {
    if (currentPage < maxPage) {
        currentPage++;
        loadPage(currentPage);
    }
});