for i in ASPIRIN HOJCOB PENCEN QQQCIG04 TETCEN UNOGIN03
do
    sed "2s/''/'$i'/g" surfaces.py >>tmp.py
    python tmp.py
    rm tmp.py
done
