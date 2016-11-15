
okay_sense_conditions="kwClip
kwRevClip
speechAlignedClip
speechRandomClip
backClip
earlyImplantClip
lateImplantClip
partialEarlyClip
partialLateClip
shiftEarlyClip
shiftLateClip
falseAlarmClip"

stop_conditions="kwClip
kwRevClip
speechAlignedClip
speechRandomClip
backClip
partialEarlyClip
shiftLateClip
falseAlarmClip"

snooze_conditions="kwClip
speechAlignedClip
speechRandomClip
backClip
partialLateClip
shiftEarlyClip
falseAlarmClip"

conditions="okay_sense stop snooze"

sourceDir=~/keyword/newdir
destDir=~/keyword/newnewdir
mkdir -p $destDir

for k in $conditions; do
  sourceDir=~/keyword/trainingWavs_$k
  eval list='$'$k\_conditions
  for c in $list; do
    echo $c
    cp $sourceDir/$c\_*$k* $destDir
    ls $destDir | wc
  done
done

