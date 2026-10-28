<!DOCTYPE html>
<html lang="zh">
  <head>
    <script>
      if (window.parent !== window && !window.location.search.includes('be_nested=1')) {
        window.stop();
      }
    </script>
    <meta charset="UTF-8">
    <link rel="icon" href="https://cdn-static.gitcode.com/static/images/logo-favicon.png">
    <link rel="dns-prefetch" href="https://res.hc-cdn.com">
    <link rel="dns-prefetch" href="https://assets-cli.s2.udesk.cn">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta content='pc,mobile' name='applicable-device'>
    <meta property="og:image" content="https://cdn-img.gitcode.com/images/og-image.png">
    <meta property="og:image:secure_url" content="https://cdn-img.gitcode.com/images/og-image.png">
    <meta name="twitter:image" content="https://cdn-img.gitcode.com/images/og-image.png">
    <meta property="og:image:width" content="1200" />
    <meta property="og:image:height" content="630" />
    <meta property="og:image:alt" content="favicon">
    <meta property="og:image:type" content="image/png">
    <meta name="baidu-site-verification" content="codeva-zQ72KLO8Ne" />
    <title>AtomGit - 全球开发者的开源社区,开源代码托管平台</title>
    <meta name="description" content="AtomGit是面向全球开发者的开源社区,包括原创博客,开源代码托管,代码协作,项目管理等。与开发者社区互动,提升您的研发效率和质量。"/>
    <meta name="keywords" content="开源社区,开源代码,AtomGit"/>
    <style type="text/css">
      .icon {
         width: 1em; height: 1em;
         vertical-align: -0.15em;
         fill: currentColor;
         overflow: hidden;
      }
    </style>
    <script type="module" crossorigin src="https://cdn-static.gitcode.com/assets/index-4e8d8e6f.js"></script>
    <link rel="modulepreload" crossorigin href="https://cdn-static.gitcode.com/assets/vendor-f43b43c4.js">
    <link rel="modulepreload" crossorigin href="https://cdn-static.gitcode.com/assets/vendor-layout-021c0a69.js">
    <link rel="stylesheet" href="https://cdn-static.gitcode.com/assets/vendor-layout-6eee6376.css">
    <link rel="stylesheet" href="https://cdn-static.gitcode.com/assets/index-ce7e0e00.css">
  </head>
  <body>
    <img src="https://cdn-static.gitcode.com/static/images/logo-favicon.png" style="position:absolute;left:-1000px;top:-1000px" alt="favicon">
    <div id="app"></div>
    
  <script data-gitcode-ssr-create-menu-fix>(() => {
  const createMenuPaths = new Set(["/create","/create/migrate","/models/create","/datasets/create","/space/create","/collections/create","/org/create","/dashboard"]);

  function toInternalHref(value) {
    if (!value) return null;
    let url;
    try {
      url = new window.URL(value, window.location.href);
    } catch {
      return null;
    }
    if (!createMenuPaths.has(url.pathname)) return null;
    return url.pathname + url.search + url.hash;
  }

  function normalizeMenuLinks() {
    const menu = window.document.getElementById('gitcode-header-create-menu');
    menu?.querySelectorAll('a.dashboard-create-menu__item').forEach(normalizeLink);
    window.document
      .querySelectorAll('a.home-header-actions__workspace')
      .forEach(normalizeLink);
  }

  function normalizeLink(link) {
    const target = toInternalHref(link.getAttribute('href'));
    if (target && link.getAttribute('href') !== target) {
      link.setAttribute('href', target);
    }
  }

  window.document.addEventListener('click', (event) => {
    if (!(event.target instanceof window.Element)) return;
    const link = event.target.closest(
      '#gitcode-header-create-menu a.dashboard-create-menu__item, ' +
        'a.home-header-actions__workspace'
    );
    if (!link) return;
    const target = toInternalHref(link.getAttribute('href'));
    if (!target) return;

    // 旧组件的 click handler 会继续使用它闭包里的绝对地址；在捕获阶段截断，
    // 再用已经归一化的站内地址完成同源导航。
    event.preventDefault();
    event.stopImmediatePropagation();
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.button !== 0) {
      window.open(target, '_blank', 'noopener');
      return;
    }
    window.location.assign(target);
  }, true);

  normalizeMenuLinks();
  if (typeof window.MutationObserver === 'function' && window.document.documentElement) {
    const observer = new window.MutationObserver(normalizeMenuLinks);
    observer.observe(window.document.documentElement, {
      subtree: true,
      childList: true,
      attributes: true,
      attributeFilter: ['href']
    });
  }
})();</script></body>
  <!-- ai 社区图标 -->
  <script type="text/javascript" src="https://cdn-static.gitcode.com/js/font_4976451_914etgmfzh8.js" defer></script>
  <script type="text/javascript" src="https://cdn-static.gitcode.com/js/font_4205092_vklvlu13jx8.js" defer></script>
  <script type="text/javascript" src="https://cdn-static.gitcode.com/js/lottie.min.js" defer></script>
  <script type="text/javascript" src="https://cdn-static.gitcode.com/js/tac/load.min.js" defer></script>
  <script type="text/javascript" src="https://cdn-static.gitcode.com/js/yunpian/riddler-sdk-0.2.2.js" defer></script>
  <script src="https://cdn-static.gitcode.com/js/yidun/yidun-captcha.js" defer></script>
  <script type="text/javascript" src="https://cdn-static.gitcode.com/js/furion.js" defer></script>
  <!-- baidu统计X -->
  <script>
    var _hmt = _hmt || [];
    _hmt.push(['_setAutoPageview', true]);
    (function() {
      var hm = document.createElement("script");
      hm.src = "https://hm.baidu.com/hm.js?62047c952451105d57bab2c4af9ce85b";
      var s = document.getElementsByTagName("script")[0];
      s.parentNode.insertBefore(hm, s);
    })();
  </script>
</html>
