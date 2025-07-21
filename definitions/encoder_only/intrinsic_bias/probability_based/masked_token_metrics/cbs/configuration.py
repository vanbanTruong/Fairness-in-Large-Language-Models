configuration = {
    'en': {
        'bert_model': 'bert-base-multilingual-uncased',
        'nationality': ['america', 'canada', 'japan', 'china', 'korea', 'england',
                        'france', 'germany', 'mexico', 'iraq', 'ireland', 'iran', 'saudi',
                        'russia', 'vietnam', 'thailand', 'australia', 'spain', 'turkey',
                        'israel', 'italy', 'egypt', 'somalia', 'india', 'brazil',
                        'colombia', 'greece', 'afghanistan', 'cuba', 'syria', 'north korea', 'pakistan', 'romania', 'switzerland', 'morocco'],
        'template_path': 'data/cbs/templates_en.txt',
        'occ_path': 'data/cbs/occ_en.txt',
        'MSK': 103
    },
    'zh':{
        'bert_model': 'bert-base-multilingual-uncased',
        'nationality': ['美国','加拿大','日本','中国','韩国','英格兰','法国','德国','墨西哥','伊拉克','爱尔兰',
                         '伊朗','沙特','俄国','越南','泰国','澳大利亚','西班牙','土耳其','以色列', '意大利','埃及',
                         '索马里','印度','巴西','哥伦比亚','希腊','阿富汗','古巴','叙利亚' ,'北朝鲜', '巴基斯坦', '罗马尼亚', '瑞士', '摩洛哥'],
        'template_path': 'data/cbs/templates_zh.txt',
        'occ_path': 'data/cbs/occ_zh.txt',
        'MSK': 103
        # 'MSK': 103
    },
}