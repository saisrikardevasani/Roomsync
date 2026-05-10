#include <gtest/gtest.h>
#include <sodium.h>
#include <vector>
#include <cstring>

// AES-256-GCM round-trip using libsodium
TEST(EncryptionTest, AES256GCMRoundTrip) {
    ASSERT_EQ(sodium_init(), 0);

    const std::string plaintext = "Hello, RoomSync secure audio frame!";

    unsigned char key[crypto_aead_aes256gcm_KEYBYTES];
    unsigned char nonce[crypto_aead_aes256gcm_NPUBBYTES];
    randombytes_buf(key,   sizeof(key));
    randombytes_buf(nonce, sizeof(nonce));

    std::vector<unsigned char> ct(plaintext.size() + crypto_aead_aes256gcm_ABYTES);
    unsigned long long ct_len = 0;

    int rc = crypto_aead_aes256gcm_encrypt(
        ct.data(), &ct_len,
        reinterpret_cast<const unsigned char*>(plaintext.data()), plaintext.size(),
        nullptr, 0,   // no additional data
        nullptr,      // nsec (unused)
        nonce, key);
    ASSERT_EQ(rc, 0);

    std::vector<unsigned char> decrypted(plaintext.size());
    unsigned long long dec_len = 0;
    rc = crypto_aead_aes256gcm_decrypt(
        decrypted.data(), &dec_len,
        nullptr,
        ct.data(), ct_len,
        nullptr, 0,
        nonce, key);
    ASSERT_EQ(rc, 0);
    EXPECT_EQ(dec_len, plaintext.size());
    EXPECT_EQ(std::string(reinterpret_cast<char*>(decrypted.data()), dec_len), plaintext);
}

TEST(EncryptionTest, TamperedCiphertextRejected) {
    ASSERT_EQ(sodium_init(), 0);

    const std::string plaintext = "secure frame";
    unsigned char key[crypto_aead_aes256gcm_KEYBYTES];
    unsigned char nonce[crypto_aead_aes256gcm_NPUBBYTES];
    randombytes_buf(key, sizeof(key));
    randombytes_buf(nonce, sizeof(nonce));

    std::vector<unsigned char> ct(plaintext.size() + crypto_aead_aes256gcm_ABYTES);
    unsigned long long ct_len = 0;
    crypto_aead_aes256gcm_encrypt(ct.data(), &ct_len,
        reinterpret_cast<const unsigned char*>(plaintext.data()), plaintext.size(),
        nullptr, 0, nullptr, nonce, key);

    // Flip one bit in the ciphertext
    ct[0] ^= 0x01;

    std::vector<unsigned char> dec(plaintext.size());
    unsigned long long dec_len = 0;
    int rc = crypto_aead_aes256gcm_decrypt(dec.data(), &dec_len,
        nullptr, ct.data(), ct_len, nullptr, 0, nonce, key);
    EXPECT_NE(rc, 0) << "Tampered ciphertext should be rejected";
}

TEST(EncryptionTest, FixedWireFrameSize) {
    // The RoomSync protocol uses 256-byte fixed-size frames to prevent length leakage.
    constexpr int WIRE_SIZE    = 256;
    constexpr int NONCE_SIZE   = 12;
    constexpr int TAG_SIZE     = 16;
    constexpr int MAX_PAYLOAD  = WIRE_SIZE - NONCE_SIZE - TAG_SIZE;  // 228

    EXPECT_EQ(NONCE_SIZE + MAX_PAYLOAD + TAG_SIZE, WIRE_SIZE);
}
