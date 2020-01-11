创建分支
git banch banchname
切换分支
git checkout banchname
创建并切换
git checkout -b banchname
查看分支状况
git log
git log --oneline --decorate --graph --all
添加文件
git add .
提交文件
git commit -m “message”
查看git的文件
git ls-tree -r master --name-only