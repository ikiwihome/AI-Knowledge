import { defineConfig } from 'vitepress'

// https://vitepress.vuejs.org/config/app-configs
export default defineConfig({
  // 网站语言设置为简体中文
  lang: 'zh-CN', 

  // 网站标题配置
  title: 'AI芯片软硬件相关技术术语', // 网站主标题
  description: '人工智能相关知识整理', // 网站描述信息

  // 主题配置
  themeConfig: {

  },

  // Markdown 配置
  markdown: {
    // 目录配置
    toc: {
      level: [2, 3, 4] // 调整为显示二级至四级标题 (对应 ## 至 ####)
    },

    // 行号显示配置
    lineNumbers: true // 显示代码块行号
  },


  // 构建配置
  vite: {
    build: {
      minify: true // 开启代码压缩
    }
  },



  // 全局头部内容
  head: [
    ['meta', { name: 'author', content: 'Cline AI' }]
  ]
})
