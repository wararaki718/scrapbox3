from create_decomposition_pipelines import use_incremental_pca, use_pca_with_function_transformer, use_truncated_svd


def main():
    texts = [
        'this is a pen',
        'that is an apple',
        'this apple is red',
        'the pen is red',
        'red is color',
        'blue is color',
        'this pen is blue',
        'the color is yellow'
    ]
    
    pca_pipe = use_pca_with_function_transformer()
    ipca_pipe = use_incremental_pca()
    tsvd_pipe = use_truncated_svd()

    pca_pipe.fit(texts)
    ipca_pipe.fit(texts)
    tsvd_pipe.fit(texts)

    print('pca_pipe result:')
    print(pca_pipe.transform(texts))
    print()

    print('ipca_pipe result:')
    print(ipca_pipe.transform(texts))
    print()

    print('tsvd_pipe result:')
    print(tsvd_pipe.transform(texts))
    print('DONE')


if __name__ == '__main__':
    main()
