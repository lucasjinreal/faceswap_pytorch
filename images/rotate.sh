d=$1
echo "roate from "$d
for szFile in $d/*.png
do 
    dd=${d}_right
    echo "save to "$dd
    if [ ! -d $dd ];then
        mkdir $dd
    fi
    convert "$szFile" -rotate 90 $dd/"$(basename "$szFile")" ; 
done
