# python analysis_tools/eval_change_caption.py  --data-dir ./s2looking_dino/S2Looking_MCI_bts_16_mobilenetv2_epo_34_Sum_138753_MIou_67721_Bleu4_71032 --out-csv s2looking_detail.csv --summary-json s2looking_summary.json
# python analysis_tools/analyze_summary.py --summary s2looking_summary.json --detail s2looking_detail.csv --out-csv s2looking_analysis.csv
# python analysis_tools/plot_summary.py   --summary s2looking_summary.json --outdir ./s2looking_cap_result
# python analysis_tools/eval_change_caption.py  --data-dir ./smars_dino/SMARS_MCI_bts_16_mobilenetv2_epo_134_Sum_178003_MIou_99420_Bleu4_78583 \
#        --out-csv smars_detail.csv --summary-json smars_summary.json
# python analysis_tools/analyze_summary.py --summary smars_summary.json --detail smars_detail.csv --out-csv smars_analysis.csv
# python analysis_tools/plot_summary.py   --summary smars_summary.json --outdir ./smars_cap_result
python analysis_tools/eval_change_caption.py  --data-dir ./LsSCD-Ex_dino/LsSCD_MCI_bts_16_mobilenetv2_epo_45_Sum_160608_MIou_76665_Bleu4_83942 --out-csv lsscd_detail.csv --summary-json lsscd_summary.json