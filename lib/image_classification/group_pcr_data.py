"""
Groups pcr data under datasets/image_classification/pcr.
For example, can be given the output of lib/image_scraping/obj_det_dataset_scraper.py, a mapping defined in this script ->
will then group the data from obj_det_dataset_scraper.py according to the mapping and store in a given directory.
"""


import os
import yaml
import shutil

import lib.file_utils as fu


SCIENTIFIC_SPECIES_NAMES_TO_COMMON_GROUP_NAMES = "SCIENTIFIC_SPECIES_NAMES_TO_COMMON_GROUP_NAMES"
OBJECT_DETECTION_TO_COMMON_GROUP_NAMES = "OBJECT_DETECTION_TO_COMMON_GROUP_NAMES"

DB_MAPS = {
    SCIENTIFIC_SPECIES_NAMES_TO_COMMON_GROUP_NAMES: {
        "surgeonfishes": ["acanthurus_achilles", "acanthurus_blochii", "acanthurus_dussumieri", "acanthurus_japonicus",
                          "acanthurus_lineatus", "acanthurus_mata", "acanthurus_nigrofuscus", "acanthurus_nigroris",
                          "acanthurus_olivaceus", "zebrasoma_veliferum", "acanthurus_pyroferus", "acanthurus_triostegus",
                          "ctenochaetus_strigosus", "zebrasoma_scopas", "paracanthurus_hepatus", "ctenochaetus_binotatus",
                          "ctenochaetus_tominiensis"],
        "unicornfishes": ["naso_vlamingii", "naso_unicornis", "naso_lituratus", "naso_brevirostris"],
        "butterflyfishes": ["chelmon_rostratus", "forcipiger_flavissimus", "heniochus_diphreutes", "hemitaurichthys_polylepis"],
        "wrasses": ["cirrhilabrus_rubrimarginatus", "pseudocheilinus_hexataenia", "stethojulis_bandanensis", "halichoeres_melanurus",
                    "bodianus_anthioides", "labroides_dimidiatus", "cirrhilabrus_rubripinnis"],
        "angelfishes": ["centropyge_flavissima", "centropyge_loricula", "centropyge_vroliki", "genicanthus_lamarck",
                        "pygoplites_diacanthus"],
        "anemonefishes": ["amphiprion_ocellaris", "premnas_biaculeatus", "amphiprion_chrysopterus", "amphiprion_clarkii",
                          "amphiprion_sebae", "amphiprion_perideraion"],
        "damselfishes": ["acanthochromis_polyacanthus", "amblyglyphidodon_ternatensis", "chromis_margaritifer",
                         "chromis_ternatensis", "chromis_viridis", "chrysiptera_parasema", "dascyllus_melanurus",
                         "pomacentrus_moluccensis", "lepidozygus_tapeinosoma", "pomacentrus_auriventris",
                         "pomacentrus_coelestis", "pomacentrus_bankanensis", "scarus_quoyi"],
        "anthias": ["pseudanthias_pleurotaenia", "pseudanthias_dispar", "pseudanthias_huchtii", "pseudanthias_squamipinnis",
                    "pseudanthias_tuka", "pseudanthias_lori", "pseudanthias_fasciatus"],
        "rabbitfishes": ["siganus_guttatus", "siganus_punctatus", "siganus_vulpinus"],
        "gobies": ["nemateleotris_magnifica", "ptereleotris_zebra"],
        "breams": ["pentapodus_emeryii", "scolopsis_bilineata"],
        "milkfish": ["chanos_chanos"],
        "hawkfishes": ["cyprinocirrhites_polyactis", "paracirrihites_arcatus", "cirrhitichthys_falco", "neocirrhites_armatus"],
        "garden_eel": ["heteroconger_hassi"],
        "soldierfishes": ["myripristis_sp"],
        "cardinalfishes": ["ostorhinchus_aureus", "zoramia_leptacanthus", "sphaeramia_nematoptera", "ostorhinchus_compressus"],
        "sweetlips": ["plectorhinchus_chaetodonoides", "plectorhinchus_vittatus"],
        "triggerfishes": ["xanthichthys_auromarginatus"],
        "fusiliers": ["caesio_teres", "pterocaesio_tile"]
    },
    OBJECT_DETECTION_TO_COMMON_GROUP_NAMES: {
        "surgeonfishes": ["acanthurus_achilles", "acanthurus_blochii", "acanthurus_dussumieri", "acanthurus_japonicus",
                          "acanthurus_lineatus", "acanthurus_mata", "acanthurus_nigrofuscus", "acanthurus_nigroris",
                          "acanthurus_olivaceus", "zebrasoma_veliferum", "acanthurus_pyroferus", "acanthurus_triostegus",
                          "ctenochaetus_strigosus", "zebrasoma_scopas", "paracanthurus_hepatus", "ctenochaetus_binotatus",
                          "ctenochaetus_tominiensis", "surgeonfish"],
        "unicornfishes": ["naso_vlamingii", "naso_unicornis", "naso_lituratus", "naso_brevirostris", "unicornfish"],
        "butterflyfishes": ["chelmon_rostratus", "forcipiger_flavissimus", "heniochus_diphreutes", "hemitaurichthys_polylepis"],
        "wrasses": ["cirrhilabrus_rubrimarginatus", "pseudocheilinus_hexataenia", "stethojulis_bandanensis", "halichoeres_melanurus",
                    "bodianus_anthioides", "labroides_dimidiatus", "cirrhilabrus_rubripinnis"],
        "angelfishes": ["centropyge_flavissima", "centropyge_loricula", "centropyge_vroliki", "genicanthus_lamarck",
                        "pygoplites_diacanthus"],
        "anemonefishes": ["amphiprion_ocellaris", "premnas_biaculeatus", "amphiprion_chrysopterus", "amphiprion_clarkii",
                          "amphiprion_sebae", "amphiprion_perideraion", "anemonefish"],
        "damselfishes": ["acanthochromis_polyacanthus", "amblyglyphidodon_ternatensis", "chromis_margaritifer",
                         "chromis_ternatensis", "chromis_viridis", "chrysiptera_parasema", "dascyllus_melanurus",
                         "pomacentrus_moluccensis", "lepidozygus_tapeinosoma", "pomacentrus_auriventris",
                         "pomacentrus_coelestis", "pomacentrus_bankanensis", "scarus_quoyi", "damselfish"],
        "anthias": ["pseudanthias_pleurotaenia", "pseudanthias_dispar", "pseudanthias_huchtii", "pseudanthias_squamipinnis",
                    "pseudanthias_tuka", "pseudanthias_lori", "pseudanthias_fasciatus", "anthia"],
        "rabbitfishes": ["siganus_guttatus", "siganus_punctatus", "siganus_vulpinus"],
        "gobies": ["nemateleotris_magnifica", "ptereleotris_zebra"],
        "breams": ["pentapodus_emeryii", "scolopsis_bilineata"],
        "milkfish": ["chanos_chanos"],
        "hawkfishes": ["cyprinocirrhites_polyactis", "paracirrihites_arcatus", "cirrhitichthys_falco", "neocirrhites_armatus"],
        "garden_eel": ["heteroconger_hassi"],
        "soldierfishes": ["myripristis_sp"],
        "cardinalfishes": ["ostorhinchus_aureus", "zoramia_leptacanthus", "sphaeramia_nematoptera", "ostorhinchus_compressus"],
        "sweetlips": ["plectorhinchus_chaetodonoides", "plectorhinchus_vittatus"],
        "triggerfishes": ["xanthichthys_auromarginatus"],
        "fusiliers": ["caesio_teres", "pterocaesio_tile", "fusilier"],
        "other": []
    }
}


if __name__ == "__main__":
    # we expect, as a hand-shake agreement, that there is a .yml config file in top level of lib/configs directory
    config_dir = os.path.join(os.pardir, 'configs', 'image_classification')
    yaml_path = os.path.join(config_dir, 'group_pcr_data.yml')
    with open(yaml_path, "r") as stream:
        config = yaml.load(stream)

    ## collect hyper parameters/args from config
    # NOTE: float() is required to parse any exponentials since YAML sends exponentials as strings
    directory_src = config["directory_src"]
    directory_dst = config["directory_dst"]
    db_map_key = config["db_map_key"]

    # initialize destination directory in case it doesn't exist
    fu.init_directory(directory_dst)

    # get database mapping
    db_map = DB_MAPS[db_map_key]

    # get class directories
    class_dirs = [d for d in os.listdir(directory_src)]

    # map the class directories according to the database mapping
    total = 0
    for class_dir in class_dirs:
        class_dir_path = os.path.join(directory_src, class_dir)

        # some datasets will have an 'other' class which will be empty in the mapping (e.g. when scraping object
        # detection data), so we just take 'other' as the bucket name
        bucket = None
        if class_dir != "other":
            for key in db_map.keys():
                if class_dir in db_map[key]:
                    bucket = key
        else:
            bucket = "other"

        # initialize directory to hold bucket category
        bucket_dir_path = os.path.join(directory_dst, bucket)
        fu.init_directory(bucket_dir_path)

        # move files to bucket
        num_class = 0
        for idx, f_path_src in enumerate([os.path.join(class_dir_path, f) for f in os.listdir(class_dir_path) if os.path.isfile(os.path.join(class_dir_path, f))]):
            basename = os.path.basename(f_path_src)
            f_path_dst = os.path.join(bucket_dir_path, "{}_{}".format(class_dir, basename))
            shutil.copyfile(f_path_src, f_path_dst)
            total += 1
            num_class += 1

        print("{}: {}".format(class_dir, num_class))

    print("Total: {}".format(total))