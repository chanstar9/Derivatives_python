# -*- coding: utf-8 -*-
"""
:Author: Chankyu Choi
:Date: 2019. 12. 17
"""
from preprocess.data_guide_imp_vol import index_imp_vol_preprocessor
from multiprocessing import Pool

file_names = ['individual_stock_options_put2']

if __name__ == '__main__':
    core_num = len(file_names)
    with Pool(core_num) as p:
        results = [p.apply_async(index_imp_vol_preprocessor, [file_name, 1]) for file_name in file_names]
        for r in results:
            r.wait()
        results = [result.get() for result in results]
        for i in range(len(results)):
            results[i].to_csv('data/{}.csv'.format(file_names[i]), index=False, encoding='utf-8')
        p.close()
        p.join()
