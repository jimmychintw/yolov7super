查看遠端 vast.ai 全部 4 個視窗

請執行以下指令並顯示結果：
```bash
ssh -p 60002 root@153.198.29.53 "
echo '╔════════════════════════ TRAIN ════════════════════════╗'
tmux capture-pane -t vast:train -p | tail -10
echo ''
echo '╔════════════════════════ CPU ══════════════════════════╗'
tmux capture-pane -t vast:cpu -p | head -10
echo ''
echo '╔════════════════════════ GPU ══════════════════════════╗'
tmux capture-pane -t vast:gpu -p | head -15
echo ''
echo '╔════════════════════════ TERMINAL ═════════════════════╗'
tmux capture-pane -t vast:terminal -p | tail -10
"
```
