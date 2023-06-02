from url_tokenisation import create_embedding_model, pad_sequences, url_to_sequence, create_vocuabulary, \
    calculate_max_length_of_sequences
from clustering import perform_clustering, reassign_labels, perform_dimensionality_reduction, plot_data


def main():
    training_urls = [
        'https://www.plaisio.gr/pc-perifereiaka/laptop-accessories/apple-accessories/apple-pontiki-magic-mouse-asirmato_3805379',
        'https://www.plaisio.gr/tilefonia-tablet/tilefona/smartphones/oneplus-nord-ce-3-lite-128gb-5g-chromatic-gray_4213858',
        'https://www.plaisio.gr/ektiposi/melania-analosima-ektiposis/photoconductors-drums/drum-samsung-r116_1978373',
        'https://www.plaisio.gr/frontida-spitiou/sideroma/preses-sideromatos/singer-presa-steamy_3003914',
        'https://www.plaisio.gr/anavathmisi-diktia/diktiaka/voip-conference/logitech-rally-bar-conference-cam-system-mini-graphite_3929051',
        'https://www.plaisio.gr/smart-tech-gadgets/ilektrokinisi/categories-krani-accessories/ninebot-vasi-kinitou-gia-scooter_3845672',
        'https://www.plaisio.gr/tileoraseis/home-audio/pikap/sony-turnable-pslx310bt_3194493',

        'https://www.kotsovolos.gr/computing/laptop-tablet-ipad/ipad-tablet-accessories/158266-apple-ipad-pro-pencil',
        'https://www.kotsovolos.gr/household-appliances/built-in/ovens/208308-electrolux-eoa9s31cx',
        'https://www.kotsovolos.gr/sound-vision/home-cinema/set-home-cinema/201559-lg-lhb625m',
        'https://www.kotsovolos.gr/gaming-gadgets/xbox-series/paixnidia-xbox-series-x/215361-nba-2k21-mamba-forever-edition',
        'https://www.kotsovolos.gr/computing/networking/modems-routers/175484-tp-link-tl-wr940n',
        'https://www.kotsovolos.gr/sound-vision/car-audio/radio-cd-autokinhtou/201558-jvc-kd-t402',
        'https://www.kotsovolos.gr/air-condition-heaters/anemistires/anemistires-dapedou/197064-morris-mfs16236',

        'https://www.ianos.gr/sto-kotetsi-tou-koko-0552440',
        'https://www.ianos.gr/thessalonikis-chalkia-ke-chalkevmata-0536746',
        'https://www.ianos.gr/atelos-iperochi-0552272',
        'https://www.ianos.gr/vasikes-arches-gia-ta-sistimata-vaseon-dedomonon-0215651',
        'https://www.ianos.gr/mia-fora-ke-mia-agapi-0552406',
        'https://www.ianos.gr/magirevontas-ellinika-0551947',
        'https://www.ianos.gr/gine-xefteri-sta-anglika-0552168',

        'https://www.moustakastoys.gr/paixnidia-oximata-exoterikou-xorou/oximata-exoterikou-xorou/petalokinita-oximata/dolu-ekskafeas-yellow-8051-401954008051/',
        'https://www.moustakastoys.gr/paixnidia-oximata-exoterikou-xorou/spitia/starplast-spiti-unicorn-magical-house-023561-401911023561/',
        'https://www.moustakastoys.gr/kalokairina/fouskotes-varkes-thalassis/intex-barka-1-atomoy-explorer-pro-100-58355np-401331058355/',
        'https://www.moustakastoys.gr/bebe-vrefika-eidi/kathismata-autokinitou/kathisma-15-36kg/joie-kathisma-aytokinitoy-trillo-ember-c1220beemb000-738718005975/',
        'https://www.moustakastoys.gr/trikikla-podilata-patinia/podilata-isorropias/barval-molto-balance-bike-me-kranos-blue-16225-401912016225/',
        'https://www.moustakastoys.gr/paixnidia-oximata-exoterikou-xorou/perissotera-paixnidia-exoterikou-xorou/tkc-set-rampa-mini-68903-401926068903/',
        'https://www.moustakastoys.gr/toublakia-kataskeves-playset/playmobil/playmobil-city-action/playmobil-duo-pack-astynomikos-kai-kallitexnis-gkrafiti-70822-840966070822/',

        'https://www.public.gr/product/books/greek-books/literature/translated-literature/to-prasino-simeiomatario/1595493',
        'https://www.public.gr/product/wearables-gadgets/smartwatches/apple-watch-se-starlight-aluminium-gps-40mm--starlight-sport-band-regular/1718852',
        'https://www.public.gr/product/sports-and-fitness/aksesoyar-gymnastikis/tsantes-sakidia/thikes-kinitoy/wantalis-thiki-kinitou-gia-to-mpratso--armband-gia-smartphones-eos-58-/MRK2073827',
        'https://www.public.gr/product/home/mikroepipla/polythrones/barcelona-poluthrona-krebati-sk-roz-kapitone-u83x106x92ek/MRK2160813',
        'https://www.public.gr/product/thermansi-klimatismos/air-condition/forita-klimatistika/inventor-magic-m3ghp290-12-12000btu-aa-psuksisthermansis-forito-klimatistiko/1697437',
        'https://www.public.gr/product/ihos/hxeia/party-speakers/forito-ixeio-akai-ss022a-x6-bluetooth-mauro/1187177',
        'https://www.public.gr/product/wearables-gadgets/e-mobility/ilektrika-patinia/ilektriko-patini-xiaomi-mi-electric-scooter-3-lite--mauro/1714861'



        'https://www.plaisio.gr/pc-perifereiaka/laptop-accessories/microsoft-surface-accessories',
        'https://www.plaisio.gr/Cart',
        'https://www.plaisio.gr/Help/PurchaseInformation/DeliveryShipping/DeliveryPricing',
        'https://www.plaisio.gr/Store-Locator',
        'https://www.plaisio.gr/tilefonia-tablet/aksesouar-gia-tablet',
        'https://www.plaisio.gr/ContactForms/ContactUs',
        'https://www.plaisio.gr/PlaisioService/Default',

        'https://www.kotsovolos.gr/mobile-phones-gps/mobile-phones/smartphones',
        'https://www.kotsovolos.gr/order-tracking/?view=footer',
        'https://www.kotsovolos.gr/pages/doseis-eksoflisi',
        'https://www.kotsovolos.gr/air-condition-heaters/aerokourtines',
        'https://www.kotsovolos.gr/ypiresies/services-episkevi-anavathmisi',
        'https://www.kotsovolos.gr/symvolaia/internet-statherh/anabathmisi',
        'https://www.kotsovolos.gr/imaging/digital-cameras',

        'https://www.ianos.gr/p/post/tropoi-pliromis',
        'https://www.ianos.gr/p/post/epistrofes',
        'https://www.ianos.gr/p/post/diaxeirish-logariasmou',
        'https://www.ianos.gr/books/epistimes',
        'https://www.ianos.gr/paichnidia/pazl',
        'https://www.ianos.gr/books/biografies/mousikon',
        'https://www.ianos.gr/chartika-eidi-dorou/organosi-grafeiou',

        'https://www.moustakastoys.gr/paixnidia-oximata-exoterikou-xorou/trampolino/',
        'https://www.moustakastoys.gr/paraggelies/parakolouthisi-paraggelias/',
        'https://www.moustakastoys.gr/to-kalathi-mou/',
        'https://www.moustakastoys.gr/paraggelies/metaforika/',
        'https://www.moustakastoys.gr/etairia/katastimata/',
        'https://www.moustakastoys.gr/etairia/epikoinonia/',
        'https://www.moustakastoys.gr/voitheia/asfaleia-paihnidion/'

        'https://www.public.gr/cat/books/paidika/theatrika-biblia',
        'https://www.public.gr/cat/computers-and-software/laptops',
        'https://www.public.gr/page/support',
        'https://www.public.gr/checkout/empty-cart',
        'https://www.public.gr/page/help/tropoi-apostolis',
        'https://www.public.gr/search-service',
        'https://www.public.gr/page/help/tropoi-pliromis'
    ]
    # 1: product url, 2: non-product url
    true_labels = [1] * int(len(training_urls) / 2) + [0] * int(len(training_urls) / 2)

    # Create vocabulary based on characters in urls
    characters, char_to_idx = create_vocuabulary(urls=training_urls)

    # Transform urls to sequences
    sequences = url_to_sequence(urls=training_urls, char_to_idx=char_to_idx)

    # Calculate max length of sequences
    max_length = calculate_max_length_of_sequences(sequences=sequences)

    # Pad sequences to have same length vectors
    padded_sequences = pad_sequences(sequences=sequences, length=max_length)

    # Create the embedding model based on our vocabulary (characters in dataset)
    model = create_embedding_model(characters=characters, length=max_length)

    # Generate character-level embeddings
    embeddings = model.predict(padded_sequences)

    # Perform k-means clustering
    predicted_labels = perform_clustering(embeddings=embeddings)

    # Reassign clustering labels with the hungarian algorithm
    predicted_labels = reassign_labels(predicted_labels=predicted_labels, true_labels=true_labels)

    # Perform dimensionality reduction to plot data to 2D space
    data = perform_dimensionality_reduction(embeddings=embeddings)

    # Plot results
    plot_data(data, true_labels=true_labels, predicted_labels=predicted_labels)


if __name__ == "__main__":
    main()
