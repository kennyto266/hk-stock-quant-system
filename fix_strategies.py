#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修復 strategies.py 文件中的語法錯誤
"""

def fix_strategies_file():
    """修復strategies.py中的語法錯誤"""
    print("修復 strategies.py 文件...")
    
    # 讀取文件
    with open('strategies.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修復未完成的字符串和f-string
    lines = content.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        line_num = i + 1
        
        # 檢查是否有未完成的字符串或f-string
        if 'print(f"' in line and not line.strip().endswith('")') and not line.strip().endswith('")\n'):
            print(f"修復第 {line_num} 行的未完成字符串: {line[:50]}...")
            # 簡單修復 - 添加結束引號和括號
            if line.strip() == 'print(f"':
                line = '        print("暴力搜索完成!")'
            elif 'print(f"' in line:
                # 嘗試完成字符串
                if '"' not in line[line.find('print(f"') + 8:]:
                    line = line + '")'
        
        # 檢查未完成的print語句
        if 'print("' in line and not line.strip().endswith('")') and not line.strip().endswith('")\n'):
            if line.strip() == 'print("':
                line = '        print("搜索完成")'
        
        fixed_lines.append(line)
    
    # 寫回文件
    fixed_content = '\n'.join(fixed_lines)
    with open('strategies.py', 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print("修復完成!")
    
    # 測試語法
    try:
        exec(compile(fixed_content, 'strategies.py', 'exec'))
        print("✅ 語法檢查通過!")
        return True
    except SyntaxError as e:
        print(f"❌ 仍有語法錯誤: {e}")
        print(f"   行號: {e.lineno}, 位置: {e.offset}")
        return False

if __name__ == "__main__":
    fix_strategies_file()