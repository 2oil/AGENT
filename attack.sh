batch_size=1
input_path='./LA/ASVspoof2019_LA_eval/flac/'
output_path='./'
adv_method1='ASV_FGSM_0003'

if [[ "$adv_method1" == CM* ]]; then
    script="gen_ad_cm.py"
elif [[ "$adv_method1" == ASV* ]]; then
    script="gen_ad_asv.py"
else
    echo "❌ Error: adv_method1 must start with 'CM' or 'ASV'"
    exit 1
fi

com="CUDA_VISIBLE_DEVICES=0 python ${script}
    --batch_size ${batch_size}
    --input_path ${input_path}
    --output_path ${output_path}
    --adv_method1 ${adv_method1}"

echo ${com}
eval ${com}