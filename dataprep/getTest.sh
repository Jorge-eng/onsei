ids="160517_07
160517_08
160606_03
160606_04
161018_01
916261_97VN760
915641_AFDP9VS
916029_6M3QU0K
930004_DQA1KBS
932739_FPD408K
932741_A57CA4G"

kws="okay_sense stop snooze"
../implementation/batchfeat.sh $1/wav $1/bin

for kw in $kws; do
  mkdir -p ~/keyword/testingWavs_kwClip_$kw/bin
  cp $1/bin/kwClip_*$kw* ~/keyword/testingWavs_kwClip_$kw/bin
done

