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

上传
列出所有远程分支
git remote
git remote -v | --verbose 列出详细信息，在每一个名字后面列出其远程url
加远程仓库
git remote add remote_name rul
e.g.  git remote add origin git@github.com:gejianwnen/IGBT_NASA.git
上传分支
git push remote_name branch_name
git push -f  强行上传
