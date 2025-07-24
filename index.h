#include <immintrin.h>
#include <vector>
#include <set>
#include "alex.h"
#include "geos/geom/Geometry.h"
#include "geos/geom/Envelope.h"
#include "geos/geom/CoordinateSequence.h"
#include "geos/geom/Point.h"
#include <geos/geom/prep/PreparedPolygon.h>

struct myPolgon {
    geos::geom::Geometry * data;
    uint64_t key;
};

struct myPage {
    std::vector<myPolgon> data;
    uint32_t minX = std::numeric_limits<uint32_t>::max();
    uint32_t minY = std::numeric_limits<uint32_t>::max();
    uint32_t maxX = 0;
    uint32_t maxY = 0;
};



int myMBRcheckSingle(uint32_t x0, uint32_t x1, uint32_t X0, uint32_t X1) {
    if (X0 > x1 || X1 < x0) return 0;
    if (X0 >= x0 && X1 <= x1) return 2;
    return 1;
}

int myMBRcheckpoint(uint32_t x, uint32_t y, geos::geom::Envelope *A) {
    if (x > A->getMaxX() || x < A->getMinX()) return 0;
    if (y > A->getMaxY() || y < A->getMinY()) return 0;
    return 1;
}

int myMBRcheck3(uint32_t x0, uint32_t x1, uint32_t y0, uint32_t y1, 
                uint32_t X0, uint32_t X1, uint32_t Y0, uint32_t Y1) {
    if (X0 > x1 || X1 < x0 || Y0 > y1 || Y1 < y0) return 0;
    if (X0 >= x0 && X1 <= x1 && Y0 >= y0 && Y1 <= y1) return 2;
    return 1;
}

int myMBRcheck4(uint32_t x0, uint32_t x1, uint32_t y0, uint32_t y1, 
                uint32_t X0, uint32_t X1, uint32_t Y0, uint32_t Y1) {
    int flag = 1;
    if (X0 > x1 || X1 < x0 || Y0 > y1 || Y1 < y0) return 0;
    if (X0 >= x0 && X1 <= x1) flag += 1; 
    if (Y0 >= y0 && Y1 <= y1) flag += 2;
    return flag; // 1 over lap or 2 cover x or 3 cover y or 4 include
}


int myMBRcheck(geos::geom::Envelope *A, geos::geom::Envelope B) {
    int include = 1;

    if (B.getMinX() > A->getMaxX() || B.getMaxX() < A->getMinX()) return 0;
    if (B.getMinY() > A->getMaxY() || B.getMaxY() < A->getMinY()) return 0;
    if (B.getMinX() >= A->getMinX() && B.getMaxX() <= A->getMaxX()) include ++;
    if (B.getMinY() >= A->getMinY() && B.getMaxY() <= A->getMaxY()) include ++;
    return include;
}

// only check non overlapped
int myMBRcheck2(geos::geom::Envelope *A, geos::geom::Envelope B) {
    if (B.getMinX() > A->getMaxX() || B.getMaxX() < A->getMinX()) return 0;
    if (B.getMinY() > A->getMaxY() || B.getMaxY() < A->getMinY()) return 0;
    return 1;
}

uint64_t absdiff(uint64_t a, uint64_t b) {
    return (a > b) ? (a - b) : (b - a);
}

static bool sortbyPolygonKey(const myPolgon &a, const myPolgon &b) {
return a.key < b.key;
}

static bool sortbyfirst(const std::pair<uint64_t, geos::geom::Geometry *> &a, 
                        const std::pair<uint64_t, geos::geom::Geometry *> &b) {
    return a.first < b.first;
}

static bool sortbyfirstPage(const std::pair<uint64_t, myPage> &a, 
                        const std::pair<uint64_t, myPage> &b) {
    return a.first < b.first;
}

static bool sortbyfirstTwoInt(const std::pair<uint64_t, uint64_t> &a, 
                        const std::pair<uint64_t, uint64_t> &b) {
    return a.first < b.first;
}

static bool sortbyfirstandsecond(const std::pair<uint64_t, uint64_t> &a, 
                        const std::pair<uint64_t, uint64_t> &b) {
    if (a.first != b.first) {
        return a.first < b.first;
    }
    return a.second < b.second;
}

static bool sortbyfirstandsecondrev(const std::pair<uint64_t, uint64_t> &a, 
                        const std::pair<uint64_t, uint64_t> &b) {
    if (a.first != b.first) {
        return a.first > b.first;
    }
    return a.second > b.second;
}

inline uint64_t BMI_e(uint64_t mask, uint64_t x){ 
	return _pdep_u64(x, mask);
}

inline uint64_t BMI_d(uint64_t mask, uint64_t z){ 
	return _pext_u64(z, mask);
}


class SpatialIndex {
    public:
        uint64_t mask_x, mask_y;
        int piece_limitation;
        std::vector<std::pair<uint64_t, uint64_t>> pieces_front;
        std::vector<std::pair<uint64_t, uint64_t>> pieces_end;
        alex::Alex<uint64_t, myPage> learnedindex;
        int page_size = 0;
        float pre_define_alpha = 1.2;
        int pre_define_beta = 64;
        int pre_define_bbeta = 128;
        SpatialIndex(uint64_t mask_x, uint64_t mask_y, int piece_limitation);
        inline uint64_t calZaddress(uint32_t x, uint32_t y);
        void bulkload(std::vector<geos::geom::Geometry *> data);
        void piecewise_front(std::pair<uint64_t, uint64_t> records[], int num_of_keys);
        void piecewise_end(std::pair<uint64_t, uint64_t> records[], int num_of_keys);
        uint32_t windowQueryCount(geos::geom::Geometry *query_window);
        void fixedPagingMethod(std::pair<uint64_t, geos::geom::Geometry *> key_value_pairs[], int num_of_keys, 
                       std::vector<myPage> & pages, std::vector<uint64_t> & keys);       
};

SpatialIndex::SpatialIndex(uint64_t mask_x, uint64_t mask_y, int piece_limitation) {
	this->mask_x = mask_x;
    this->mask_y = mask_y;
    this->piece_limitation = piece_limitation;
}

inline uint64_t SpatialIndex::calZaddress(uint32_t x, uint32_t y) {
	return BMI_e(mask_x, x) | BMI_e(mask_y, y);
}

void SpatialIndex::fixedPagingMethod(std::pair<uint64_t, geos::geom::Geometry *> key_value_pairs[], int num_of_keys, 
                       std::vector<myPage> & pages, std::vector<uint64_t> & keys) {
    myPage mp;
    myPolgon mpo;
    uint64_t key;
    mpo.data = key_value_pairs[0].second;
    key = key_value_pairs[0].first;
    const geos::geom::Envelope *envelope = key_value_pairs[0].second->getEnvelopeInternal();
    mp.minX = envelope->getMinX();
    mp.minY = envelope->getMinY();
    mp.maxX = envelope->getMaxX();
    mp.maxY = envelope->getMaxY();
    mpo.key = key;
    mp.data.emplace_back(mpo);
    double area = (((double) (mp.maxX - mp.minX))/std::numeric_limits<uint32_t>::max()) * 
                   (((double)(mp.maxY - mp.minY))/std::numeric_limits<uint32_t>::max());
    //std::cout << mp.minX << " " << mp.minY << " " << mp.maxX << " " << mp.maxY << std::endl;
    for (int i = 1; i < num_of_keys; i++) {
        const geos::geom::Envelope *envelope = key_value_pairs[i].second->getEnvelopeInternal();
        mpo.data = key_value_pairs[i].second;
        mpo.key = key_value_pairs[i].first;
        //std::cout << mpo.minX << " " << mpo.minY << " " <<  mpo.maxX << " " << mpo.maxY << std::endl;
        uint32_t tmp_minX = std::min(mp.minX, (uint32_t)envelope->getMinX());
        uint32_t tmp_minY = std::min(mp.minY, (uint32_t)envelope->getMinY());
        uint32_t tmp_maxX = std::max(mp.maxX, (uint32_t)envelope->getMaxX());
        uint32_t tmp_maxY = std::max(mp.maxY, (uint32_t)envelope->getMaxY());
        double tmp_area = (((double) (tmp_maxX - tmp_minX))/std::numeric_limits<uint32_t>::max()) * 
                   (((double)(tmp_maxY - tmp_minY))/std::numeric_limits<uint32_t>::max());
        //if (mp.data.size() < 128 || mp.data.size() >= 128 && mp.data.size() < 256 && area * 1.2 >= tmp_area) {
        if (mp.data.size() < 128 || mp.data.size() >= 128 && mp.data.size() < 256 && area * 1.2 >= tmp_area) {
            mp.data.emplace_back(mpo);
            mp.minX = std::min(mp.minX, (uint32_t)envelope->getMinX());
            mp.minY = std::min(mp.minY, (uint32_t)envelope->getMinY());
            mp.maxX = std::max(mp.maxX, (uint32_t)envelope->getMaxX());
            mp.maxY = std::max(mp.maxY, (uint32_t)envelope->getMaxY());
        } else {
            pages.emplace_back(mp);
            keys.emplace_back(key);
            mp.data.clear();
            mp.data.emplace_back(mpo);
            key = key_value_pairs[i].first;
            mp.minX = envelope->getMinX();
            mp.minY = envelope->getMinY();
            mp.maxX = envelope->getMaxX();
            mp.maxY = envelope->getMaxY();
            area = (((double) (mp.maxX - mp.minX))/std::numeric_limits<uint32_t>::max()) * 
                    (((double)(mp.maxY - mp.minY))/std::numeric_limits<uint32_t>::max());
        }
    }
    pages.emplace_back(mp);
    keys.emplace_back(key);
    page_size = pages.size();
}

void SpatialIndex::bulkload(std::vector<geos::geom::Geometry *> data) {
    int num_of_keys = data.size();
    std::pair<uint64_t, geos::geom::Geometry *> *key_value_pairs = new std::pair<uint64_t, geos::geom::Geometry *>[num_of_keys];
    //std::pair<uint64_t, uint64_t> *values_front = new std::pair<uint64_t, uint64_t>[num_of_keys];
    //std::pair<uint64_t, uint64_t> *values_end = new std::pair<uint64_t, uint64_t>[num_of_keys];
    for (int i = 0; i < num_of_keys; i++) {
        const geos::geom::Envelope *envelope = data[i]->getEnvelopeInternal();
        uint32_t minX = envelope->getMinX();
        uint32_t minY = envelope->getMinY();
        uint32_t maxX = envelope->getMaxX();
        uint32_t maxY = envelope->getMaxY();
        uint64_t z_start = calZaddress(minX, minY);
        uint64_t z_end = calZaddress(maxX, maxY);
        uint64_t mid = (z_end + z_start) / 2;
        uint64_t key = z_start;
        const geos::geom::Polygon* polygon = dynamic_cast<const geos::geom::Polygon*>(data[i]);

        const geos::geom::LineString* lineString = polygon->getExteriorRing();
        std::unique_ptr<geos::geom::CoordinateSequence> coordSeq = lineString->getCoordinates();
        const geos::geom::CoordinateSequence* coords = coordSeq.get();
        for (size_t i = 0; i < coords->size(); ++i) {
            const geos::geom::Coordinate& coord = coords->getAt(i);
            uint64_t z_value = calZaddress(coord.x, coord.y);
            key = findmid(z_value, key, mid);
        }

        key_value_pairs[i].first = key;
        key_value_pairs[i].second = data[i];
    }

    std::sort(key_value_pairs, key_value_pairs + num_of_keys, sortbyfirst);
    std::vector<myPage> pages;
    std::vector<uint64_t> keys;
    fixedPagingMethod(key_value_pairs, num_of_keys, pages, keys);
    //HeuristicPagingMethod(key_value_pairs, num_of_keys, pages, keys);
    std::pair<uint64_t, myPage> * key_value_pairs2 = new std::pair<uint64_t, myPage>[keys.size()];
    std::pair<uint64_t, uint64_t> *values_front = new std::pair<uint64_t, uint64_t>[keys.size()];
    std::pair<uint64_t, uint64_t> *values_end = new std::pair<uint64_t, uint64_t>[keys.size()];
    for (int i = 0; i < keys.size(); i++) {
        key_value_pairs2[i].first = keys[i];
        key_value_pairs2[i].second = pages[i];
        values_front[i].first = keys[i];
        values_front[i].second = calZaddress(pages[i].maxX, pages[i].maxY);
        //std::cout << keys[i] << " " << calZaddress(pages[i].maxX, pages[i].maxY) << std::endl;
        values_end[i].first = keys[i];
        values_end[i].second = calZaddress(pages[i].minX, pages[i].minY);
    }

    piecewise_front(values_front, keys.size());
    piecewise_end(values_end, keys.size());
    delete[] values_front;
    delete[] values_end;
    //std::sort(key_value_pairs2, key_value_pairs2 + keys.size(), sortbyfirstPage);
    learnedindex.bulk_load(key_value_pairs2, keys.size());
    delete[] key_value_pairs;
    delete[] key_value_pairs2;
    auto it_start = learnedindex.begin();
    auto it_end = learnedindex.end();
    // Generate the MBR in each leaf node (data node)
    for (auto it = it_start; it != it_end; it.it_update_mbr()) {
    }
}

void SpatialIndex::piecewise_front(std::pair<uint64_t, uint64_t> records[], int num_of_keys) {
    std::sort(records, records + num_of_keys, sortbyfirstandsecond);
    uint64_t current_count = 0;
    uint64_t current_z = 0;
    uint64_t current_zmax = 0;
    uint64_t pre_zmax = 0;
    for (int i = 0; i < num_of_keys; i++) {
        if (current_count == 0) {
            current_z = records[i].first;
            current_zmax = records[i].second;
        } else {
            if (records[i].second > current_zmax * 1.2) {
                if (current_zmax > pre_zmax) {
                    pieces_front.emplace_back(current_zmax, current_z);
                    pre_zmax = current_zmax;
                }
                current_count = 0;
                current_z = records[i].first;
                current_zmax = records[i].second;
            } else {
                current_zmax = std::max(records[i].second, current_zmax);
            }
        }
        current_count ++;
        if (current_count == piece_limitation) {
            //std::cout << current_zmax << " " << pre_zmax << std::endl;
            if (current_zmax > pre_zmax) {
                pieces_front.emplace_back(current_zmax, current_z);
                pre_zmax = current_zmax;
            }
            current_count = 0;
        }
    }
    if(current_count > 0){
        if (current_zmax > pre_zmax) {
            pieces_front.emplace_back(current_zmax, current_z);
        }
    }
    // std::cout << pieces_front.size() << std::endl;
}

// pair: key, min
void SpatialIndex::piecewise_end(std::pair<uint64_t, uint64_t> records[], int num_of_keys) {
    std::sort(records, records + num_of_keys, sortbyfirstandsecondrev);
    uint64_t current_count = 0;
    uint64_t current_z = 0;
    uint64_t current_zmin = 0;
    uint64_t pre_zmin = 18446744073709551615ULL;
    for (int i = 0; i < num_of_keys; i++) {
        if (current_count == 0) {
            current_z = records[i].first;
            current_zmin = records[i].second;
        } else {
            current_zmin = std::min(records[i].second, current_zmin);
        }
        current_count ++;
        if (current_count == piece_limitation) {
            if (current_zmin < pre_zmin) {
                pieces_end.emplace_back(current_zmin, current_z);
                pre_zmin = current_zmin;
            }
            current_count = 0;
        }
    }
    if(current_count > 0){
        if (current_zmin < pre_zmin) {
            pieces_end.emplace_back(current_zmin, current_z);
        }
    }
    std::reverse(pieces_end.begin(), pieces_end.end());
}

uint32_t SpatialIndex::windowQueryCount(geos::geom::Geometry *query_window) {
    uint32_t res = 0;
    geos::geom::Envelope env_query_window = *query_window->getEnvelopeInternal();
     geos::geom::prep::PreparedPolygon preparedPolygon(query_window);
    uint32_t minX = env_query_window.getMinX();
    uint32_t minY = env_query_window.getMinY();
    uint32_t maxX = env_query_window.getMaxX();
    uint32_t maxY = env_query_window.getMaxY();
    uint64_t q_start = calZaddress(minX, minY);
    uint64_t q_end = calZaddress(maxX, maxY);
    //std::set<geos::geom::Geometry *> mySet;
    auto it = std::lower_bound(pieces_front.begin(), pieces_front.end(), std::make_pair(q_start, -1), sortbyfirstTwoInt);
	if (it == pieces_front.end()) it--;
    std::cout << q_start << std::endl;
    q_start = pieces_front[it - pieces_front.begin()].second;
    auto it_start = learnedindex.find_last_no_greater_than(q_start);

    it = std::upper_bound(pieces_end.begin(), pieces_end.end(), std::make_pair(q_end, -1), sortbyfirstTwoInt);
    if (it != pieces_end.begin()) it--;
    q_end = pieces_end[it - pieces_end.begin()].second;
    //std::cout << q_start << "  " << q_end << std::endl;
    int flag = -1;
    geos::geom::GeometryFactory::Ptr factory = geos::geom::GeometryFactory::create();
    for (; it_start.cur_leaf_ != nullptr && it_start.key() <= q_end; it_start.it_check_mbr(&env_query_window, q_end)) {
        flag = myMBRcheck4(minX, maxX, minY, maxY, 
                           it_start.payload().minX, it_start.payload().maxX, it_start.payload().minY, it_start.payload().maxY);
        if (flag == 0) {
            ;
        } else {
            for (int i = 0; i < it_start.payload().data.size(); i++) {
                uint32_t x = BMI_d(mask_x, it_start.payload().data[i].key);
                uint32_t y = BMI_d(mask_y, it_start.payload().data[i].key);
                std::unique_ptr<geos::geom::Point> point(factory->createPoint(geos::geom::Coordinate(x, y)));
                if (preparedPolygon.contains(point.get())) {
                    res += 1;
                } else {
                    geos::geom::Geometry *payload = it_start.payload().data[i].data;
                    if (preparedPolygon.intersects(payload)) {
                        res += 1;
                    }
                }
            }
        }
    }
    return res;
}
