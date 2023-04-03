if [ "$1" = "flush" ] && [ -f "template.py" ]; then
    read -p "T1 & T2 & T3 & T4 will be flushed. Are you sure? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Okay, bye!!"
        [[ "$0" = "$ZH_SOURCE" ]] && exit 1
    fi
    cp "template.py" "T1.py"
    cp "template.py" "T2.py"
    cp "template.py" "T3.py"
    cp "template.py" "T4.py"
    echo "Done!"
elif test -f "T$1.py"; then
    python3 T$1.py
else
    echo "Wrong commmand or maybe in the wrong place."
fi
