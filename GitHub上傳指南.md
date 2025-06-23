# 港股量化分析系統 - GitHub 上傳指南

## 📋 準備工作

### 1. 確保已安裝 Git
```bash
# 檢查 Git 版本
git --version

# 如果未安裝，從官網下載：https://git-scm.com/
```

### 2. 配置 Git（首次使用）
```bash
git config --global user.name "您的用戶名"
git config --global user.email "您的郵箱"
```

## 🚀 上傳步驟

### 步驟 1: 初始化本地 Git 倉庫
在項目根目錄打開命令行（PowerShell 或 Git Bash）：

```bash
# 初始化 Git 倉庫
git init

# 添加所有文件到暫存區
git add .

# 創建首次提交
git commit -m "初始化港股量化分析系統"
```

### 步驟 2: 在 GitHub 創建遠程倉庫

1. 登錄 [GitHub](https://github.com)
2. 點擊右上角 "+" → "New repository"
3. 填寫倉庫信息：
   - **Repository name**: `港股量化分析系統` 或 `hk-stock-quant-system`
   - **Description**: `港股量化分析與策略回測系統 - HK Stock Quantitative Analysis System`
   - **Visibility**: 選擇 Public 或 Private
   - **不要** 勾選 "Add a README file"（我們已經有了）
4. 點擊 "Create repository"

### 步驟 3: 連接本地倉庫到 GitHub

```bash
# 添加遠程倉庫（替換 YOUR_USERNAME 為您的 GitHub 用戶名）
git remote add origin https://github.com/YOUR_USERNAME/港股量化分析系統.git

# 或者使用英文名稱
git remote add origin https://github.com/YOUR_USERNAME/hk-stock-quant-system.git

# 推送到 GitHub 主分支
git branch -M main
git push -u origin main
```

### 步驟 4: 驗證上傳成功

1. 刷新 GitHub 頁面
2. 確認所有文件已上傳
3. 檢查 README.md 是否正確顯示

## 📁 已上傳的主要文件

### 核心系統文件
- ✅ `enhanced_interactive_dashboard.py` - 主要儀表板
- ✅ `run_analysis.py` - 運行分析腳本
- ✅ `strategies.py` - 策略實現
- ✅ `data_handler.py` - 數據處理
- ✅ `config.py` - 配置管理

### 策略相關文件
- ✅ `north_south_flow_strategies.py` - 南北水策略
- ✅ `integrated_north_south_strategies.py` - 整合策略
- ✅ `enhanced_strategy_optimizer.py` - 策略優化器

### 文檔文件
- ✅ `README.md` - 項目說明
- ✅ `港股量化分析系統_NonPrice數據整合指南.md` - 數據整合指南
- ✅ `requirements.txt` - 依賴清單

### 配置文件
- ✅ `.gitignore` - Git 忽略文件
- ✅ `GitHub上傳指南.md` - 本指南

## 🔧 後續管理

### 更新代碼到 GitHub
```bash
# 添加修改的文件
git add .

# 提交修改
git commit -m "描述您的修改內容"

# 推送到 GitHub
git push
```

### 創建分支（開發新功能時）
```bash
# 創建並切換到新分支
git checkout -b feature/新功能名稱

# 開發完成後推送分支
git push -u origin feature/新功能名稱

# 在 GitHub 上創建 Pull Request
```

### 克隆到其他電腦
```bash
git clone https://github.com/YOUR_USERNAME/hk-stock-quant-system.git
cd hk-stock-quant-system
pip install -r requirements.txt
```

## 📝 提交信息建議

使用清晰的提交信息：
- `feat: 添加新功能` - 新功能
- `fix: 修復bug` - 錯誤修復
- `docs: 更新文檔` - 文檔修改
- `style: 代碼格式` - 格式調整
- `refactor: 重構代碼` - 代碼重構
- `test: 添加測試` - 測試相關
- `chore: 其他修改` - 其他維護

## 🛡️ 隱私保護

`.gitignore` 文件已配置，以下內容不會上傳：
- 🚫 虛擬環境文件夾 (`venv_310/`, `.venv/`)
- 🚫 輸出數據 (`data_output/`, `港股輸出/`)
- 🚫 日誌文件 (`*.log`, `logs/`)
- 🚫 緩存文件 (`__pycache__/`, `.cache/`)
- 🚫 大型文件 (`*.exe`, `*.whl`)
- 🚫 IDE 配置 (`.vscode/`, `.cursor/`)

## ❗ 注意事項

1. **第一次推送可能需要輸入 GitHub 用戶名和密碼**
2. **如果啟用了兩步驗證，需要使用 Personal Access Token**
3. **確保不要上傳包含 API 密鑰的文件**
4. **大型文件（>100MB）需要使用 Git LFS**

## 🔗 有用的 GitHub 功能

- **Issues**: 追蹤 bug 和功能請求
- **Wiki**: 創建詳細的項目文檔
- **Actions**: 設置自動化測試和部署
- **Releases**: 發布版本標記
- **Projects**: 項目管理看板

---

🎉 **完成！** 您的港股量化分析系統現在已在 GitHub 上了！ 